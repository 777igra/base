//! How [`BatchEncoder`] packs frames into blobs and transactions, and the blob override.

use std::collections::BTreeSet;

use alloy_primitives::B256;
use base_batcher_encoder::{BatchPipeline, DaType, EncoderConfig, StepResult, SubmissionPayload};
use rstest::rstest;

use crate::common::{
    BlockFixture, EncoderFixture, MULTI_FRAME_PAYLOAD, SMALL_FRAME_SIZE, SharedBlob,
    SubmissionFixture, THREE_BLOB_PAYLOAD,
};

/// Small frames are packed together in one blob rather than one blob each.
#[test]
fn frames_are_packed_into_one_blob() {
    let config = EncoderConfig { max_frame_size: SMALL_FRAME_SIZE, ..EncoderConfig::default() };
    let fixture = EncoderFixture::new(config);
    let mut encoder = fixture.encoder();
    encoder.add_block(BlockFixture::block(B256::ZERO, 1, MULTI_FRAME_PAYLOAD)).unwrap();

    let submissions = encoder.encode_and_drain().unwrap();

    assert_eq!(submissions.len(), 1);
    assert_eq!(submissions[0].blob_count(), 1);
    assert!(submissions[0].frame_count() > 2, "{} frames", submissions[0].frame_count());
}

/// A transaction carries `max_blobs_per_tx` blobs while blobs are ready, and that cap cuts
/// transactions, not channels: one channel spans several transactions.
#[rstest]
#[case(1)]
#[case(2)]
fn transactions_carry_up_to_max_blobs_per_tx(#[case] max_blobs_per_tx: usize) {
    let config = EncoderConfig { max_blobs_per_tx, ..EncoderConfig::default() };
    let fixture = EncoderFixture::new(config);
    let mut encoder = fixture.encoder();
    encoder.add_block(BlockFixture::block(B256::ZERO, 1, THREE_BLOB_PAYLOAD)).unwrap();

    let submissions = encoder.encode_and_drain().unwrap();

    assert!(submissions.len() >= 2, "{} submissions", submissions.len());
    assert_eq!(submissions[0].blob_count(), max_blobs_per_tx);
    assert_eq!(fixture.derive(&submissions).len(), 1);
}

/// A blob takes the tail of a closed channel and the start of the next one: channels are
/// packed across blob boundaries, and derivation still reads them apart.
#[test]
fn channels_are_packed_across_a_blob_boundary() {
    let config = EncoderConfig { compressed_size_target: Some(1), ..EncoderConfig::default() };
    let fixture = EncoderFixture::new(config);
    let mut encoder = fixture.encoder();
    let shared = SharedBlob::encode(&mut encoder);

    let SubmissionPayload::Blobs(blobs) = shared.packed.payload() else {
        panic!("expected a blob submission");
    };
    let channel_ids: BTreeSet<_> = blobs[0].frames().iter().map(|frame| frame.id).collect();
    assert_eq!(channel_ids.len(), 2, "the blob carries frames of both channels");

    let mut submissions = vec![shared.first, shared.packed];
    submissions.extend(encoder.encode_and_drain().unwrap());
    let derived = fixture.derive(&submissions);
    assert_eq!(derived.len(), 2);
    assert_eq!(derived.concat(), BlockFixture::batches(&shared.blocks));
}

/// While the blob override is active, a calldata encoder emits blobs, and a retry keeps the
/// DA type its submission was built with.
#[test]
fn blob_override_switches_a_calldata_encoder_to_blobs() {
    let config = EncoderConfig { da_type: DaType::Calldata, ..EncoderConfig::default() };
    let fixture = EncoderFixture::new(config);
    let mut encoder = fixture.encoder();
    encoder.add_block(BlockFixture::block(B256::ZERO, 1, 0)).unwrap();
    assert_eq!(encoder.step().unwrap(), StepResult::BlockEncoded);
    encoder.flush().unwrap();

    encoder.set_blob_override(true);
    let submission = encoder.next_submission().expect("submission under the override");
    assert_eq!(submission.da_type(), DaType::Blob);

    encoder.requeue(submission.id);
    encoder.set_blob_override(false);
    let retry = encoder.next_submission().expect("retry after the override");
    assert_eq!(retry.da_type(), DaType::Blob);
}

/// A blob built under the override is retried on its own, never packed with the calldata
/// built once the override is off, since a transaction carries one DA type.
#[test]
fn a_blob_retry_is_not_packed_with_calldata() {
    let config = EncoderConfig { da_type: DaType::Calldata, ..EncoderConfig::default() };
    let fixture = EncoderFixture::new(config);
    let mut encoder = fixture.encoder();
    let blocks = BlockFixture::chain(2, 0);
    encoder.set_blob_override(true);
    encoder.add_block(blocks[0].clone()).unwrap();
    let blob = encoder.encode_and_drain().unwrap().remove(0);
    encoder.set_blob_override(false);
    encoder.add_block(blocks[1].clone()).unwrap();
    let calldata = encoder.encode_and_drain().unwrap().remove(0);

    encoder.requeue(blob.id);
    encoder.requeue(calldata.id);

    let retry = encoder.next_submission().expect("the blob retry");
    assert_eq!(retry.da_type(), DaType::Blob);
    assert_eq!(SubmissionFixture::frames(&retry), SubmissionFixture::frames(&blob));
    let retry = encoder.next_submission().expect("the calldata retry");
    assert_eq!(retry.da_type(), DaType::Calldata);
    assert_eq!(SubmissionFixture::frames(&retry), SubmissionFixture::frames(&calldata));
}
