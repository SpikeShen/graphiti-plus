"""
0. Extract audio from video and transcribe via AWS Transcribe.

This step is already done — the transcript JSON is pre-generated.
Run this script to re-transcribe if needed.

Usage:
    cd graphiti
    examples/audio-extraction/run.sh 0
"""

import json
import os
import subprocess
import sys
import time
import uuid

import boto3

sys.path.insert(0, os.path.dirname(__file__))
from common import REGION, SCRIPT_DIR, TRANSCRIPT_PATH, VIDEO_PATH

S3_BUCKET = os.environ.get('MULTIMODAL_ASSET_BUCKET', 'graphiti-multimodal-assets-poc')
S3_PREFIX = 'audio-transcribe'


def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video using ffmpeg (16kHz mono WAV for Transcribe)."""
    print(f'Extracting audio: {video_path} → {audio_path}', flush=True)
    subprocess.run(
        ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
         '-ar', '16000', '-ac', '1', audio_path, '-y'],
        check=True, capture_output=True,
    )
    size_kb = os.path.getsize(audio_path) / 1024
    print(f'  Audio extracted: {size_kb:.0f} KB', flush=True)


def upload_and_transcribe(audio_path: str) -> dict:
    """Upload audio to S3 and run AWS Transcribe."""
    s3 = boto3.client('s3', region_name=REGION)
    transcribe = boto3.client('transcribe', region_name=REGION)

    audio_key = f'{S3_PREFIX}/{os.path.basename(audio_path)}'
    print(f'Uploading to s3://{S3_BUCKET}/{audio_key}', flush=True)
    s3.upload_file(audio_path, S3_BUCKET, audio_key)

    job_name = f'audio-extraction-{uuid.uuid4().hex[:8]}'
    output_key = f'{S3_PREFIX}/transcript-{job_name}.json'

    print(f'Starting transcription job: {job_name}', flush=True)
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode='zh-CN',
        MediaFormat='wav',
        Media={'MediaFileUri': f's3://{S3_BUCKET}/{audio_key}'},
        OutputBucketName=S3_BUCKET,
        OutputKey=output_key,
    )

    # Poll until complete
    while True:
        resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = resp['TranscriptionJob']['TranscriptionJobStatus']
        if status == 'COMPLETED':
            print(f'  Transcription completed.', flush=True)
            break
        elif status == 'FAILED':
            reason = resp['TranscriptionJob'].get('FailureReason', 'unknown')
            raise RuntimeError(f'Transcription failed: {reason}')
        print(f'  Status: {status}, waiting...', flush=True)
        time.sleep(5)

    # Download result
    local_path = os.path.join(SCRIPT_DIR, f'transcript-{job_name}.json')
    s3.download_file(S3_BUCKET, output_key, local_path)
    with open(local_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    # Also save as the canonical transcript path
    with open(TRANSCRIPT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'  Transcript saved to: {TRANSCRIPT_PATH}', flush=True)

    transcript_text = result['results']['transcripts'][0]['transcript']
    print(f'  Text length: {len(transcript_text)} chars', flush=True)
    print(f'  Preview: {transcript_text[:200]}...', flush=True)

    # Cleanup temp audio
    os.remove(audio_path)
    return result


def main():
    audio_path = os.path.join(SCRIPT_DIR, 'temp_audio.wav')
    extract_audio(VIDEO_PATH, audio_path)
    upload_and_transcribe(audio_path)
    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
