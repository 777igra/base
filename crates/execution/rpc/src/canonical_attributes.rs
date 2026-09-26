//! Payload attributes that rebuild a canonical block.

use alloy_consensus::BlockHeader;
use alloy_eips::eip2718::Encodable2718;
use alloy_primitives::B64;
use base_common_chains::Upgrades;
use base_common_consensus::{EIP1559ParamError, HoloceneExtraData, JovianExtraData};
use base_common_rpc_types_engine::BasePayloadAttributes;
use reth_primitives_traits::{Block, BlockBody};

/// Derives the `debug_executePayload` attributes for a canonical block.
///
/// Must stay identical to the prover host's `payload_attributes_from_l2_block`
/// (`base-proof-host`), otherwise prebuilt witnesses are keyed under a different payload ID
/// than the prover requests and are never served.
#[derive(Debug)]
pub struct CanonicalPayloadAttributes;

impl CanonicalPayloadAttributes {
    /// Returns the payload attributes that rebuild `block` on top of its parent.
    pub fn from_block<B: Block>(
        block: &B,
        chain_spec: &impl Upgrades,
    ) -> Result<BasePayloadAttributes, EIP1559ParamError> {
        let header = block.header();
        let timestamp = header.timestamp();

        let mut attributes = BasePayloadAttributes::default();
        attributes.payload_attributes.timestamp = timestamp;
        attributes.payload_attributes.prev_randao = header.mix_hash().unwrap_or_default();
        attributes.payload_attributes.suggested_fee_recipient = header.beneficiary();
        attributes.payload_attributes.parent_beacon_block_root = header.parent_beacon_block_root();
        attributes.payload_attributes.withdrawals =
            block.body().withdrawals().map(|withdrawals| withdrawals.to_vec());
        attributes.transactions =
            Some(block.body().transactions().iter().map(|tx| tx.encoded_2718().into()).collect());
        attributes.no_tx_pool = Some(true);
        attributes.gas_limit = Some(header.gas_limit());

        // Holocene and Jovian extra data both start with a version byte followed by the
        // `denominator || elasticity` pair, which is exactly the `eip1559Params` encoding.
        if chain_spec.is_jovian_active_at_timestamp(timestamp) {
            let (_, _, min_base_fee) = JovianExtraData::decode(header.extra_data())?;
            attributes.eip_1559_params = Some(B64::from_slice(&header.extra_data()[1..9]));
            attributes.min_base_fee = Some(min_base_fee);
        } else if chain_spec.is_holocene_active_at_timestamp(timestamp) {
            HoloceneExtraData::decode(header.extra_data())?;
            attributes.eip_1559_params = Some(B64::from_slice(&header.extra_data()[1..9]));
        }

        Ok(attributes)
    }
}

#[cfg(test)]
mod tests {
    use alloy_consensus::{BlockBody as AlloyBlockBody, Header, Sealable};
    use alloy_eips::eip4895::{Withdrawal, Withdrawals};
    use alloy_primitives::{Address, B256, Bytes, address};
    use base_common_consensus::{BaseBlock, BaseTransactionSigned, TxDeposit};
    use base_execution_chainspec::{BaseChainSpec, BaseChainSpecBuilder};

    use super::*;

    const TIMESTAMP: u64 = 1;

    fn holocene() -> BaseChainSpec {
        BaseChainSpecBuilder::base_mainnet().holocene_activated().build()
    }

    fn jovian() -> BaseChainSpec {
        BaseChainSpecBuilder::base_mainnet().jovian_activated().build()
    }

    fn jovian_extra_data(min_base_fee: u64) -> Bytes {
        let mut extra_data = vec![1, 0, 0, 0, 250, 0, 0, 0, 6];
        extra_data.extend_from_slice(&min_base_fee.to_be_bytes());
        extra_data.into()
    }

    fn block(extra_data: Bytes, withdrawals: Vec<Withdrawal>) -> BaseBlock {
        let deposit = BaseTransactionSigned::Deposit(
            TxDeposit {
                source_hash: B256::repeat_byte(1),
                from: Address::repeat_byte(2),
                gas_limit: 1_000_000,
                ..Default::default()
            }
            .seal_slow(),
        );
        BaseBlock {
            header: Header {
                parent_hash: B256::repeat_byte(4),
                timestamp: TIMESTAMP,
                mix_hash: B256::repeat_byte(5),
                beneficiary: address!("4200000000000000000000000000000000000011"),
                parent_beacon_block_root: Some(B256::repeat_byte(3)),
                gas_limit: 30_000_000,
                extra_data,
                ..Default::default()
            },
            body: AlloyBlockBody {
                transactions: vec![deposit],
                ommers: vec![],
                withdrawals: Some(Withdrawals::new(withdrawals)),
            },
        }
    }

    #[test]
    fn holocene_block_copies_header_body_and_extra_data_params() {
        let block =
            block(Bytes::from_static(&[0, 0, 0, 0, 250, 0, 0, 0, 6]), vec![Withdrawal::default()]);

        let attributes = CanonicalPayloadAttributes::from_block(&block, &holocene()).unwrap();

        assert_eq!(attributes.eip_1559_params, Some(B64::from([0, 0, 0, 250, 0, 0, 0, 6])));
        assert_eq!(attributes.min_base_fee, None);
        assert_eq!(attributes.payload_attributes.timestamp, TIMESTAMP);
        assert_eq!(attributes.payload_attributes.prev_randao, block.header.mix_hash);
        assert_eq!(attributes.payload_attributes.suggested_fee_recipient, block.header.beneficiary);
        assert_eq!(
            attributes.payload_attributes.parent_beacon_block_root,
            block.header.parent_beacon_block_root
        );
        assert_eq!(attributes.payload_attributes.withdrawals, Some(vec![Withdrawal::default()]));
        assert_eq!(
            attributes.transactions,
            Some(vec![block.body.transactions[0].encoded_2718().into()])
        );
        assert_eq!(attributes.no_tx_pool, Some(true));
        assert_eq!(attributes.gas_limit, Some(30_000_000));
    }

    #[test]
    fn jovian_block_copies_extra_data_params_and_min_base_fee() {
        let block = block(jovian_extra_data(1_000_000), vec![]);

        let attributes = CanonicalPayloadAttributes::from_block(&block, &jovian()).unwrap();

        assert_eq!(attributes.eip_1559_params, Some(B64::from([0, 0, 0, 250, 0, 0, 0, 6])));
        assert_eq!(attributes.min_base_fee, Some(1_000_000));
    }

    #[test]
    fn extra_data_of_the_wrong_fork_is_rejected() {
        let holocene_extra_data = Bytes::from_static(&[0, 0, 0, 0, 250, 0, 0, 0, 6]);

        assert_eq!(
            CanonicalPayloadAttributes::from_block(&block(holocene_extra_data, vec![]), &jovian()),
            Err(EIP1559ParamError::InvalidExtraDataLength)
        );
        assert_eq!(
            CanonicalPayloadAttributes::from_block(
                &block(jovian_extra_data(1), vec![]),
                &holocene()
            ),
            Err(EIP1559ParamError::InvalidExtraDataLength)
        );
    }

    #[test]
    fn payload_id_survives_the_prover_json_rpc_hop() {
        let block = block(jovian_extra_data(7), vec![]);
        let attributes = CanonicalPayloadAttributes::from_block(&block, &jovian()).unwrap();

        let json = serde_json::to_string(&attributes).unwrap();
        let decoded: BasePayloadAttributes = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, attributes);
        assert_eq!(
            decoded.payload_id(&block.header.parent_hash, 3),
            attributes.payload_id(&block.header.parent_hash, 3)
        );
    }
}
