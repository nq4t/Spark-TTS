"""
SparkTTS Inference CLI Interface
SparkAudio - Xinsheng Wang - AcTePuKc - Jay Moore (nq4t)

Provides command line interface to Spark Inference.
Requires nltk and punkt.

Provides a way of using inference via the command line by arguments.
Was designed to be used with sparkdirector.sh - usually in a text editor with integrated terminal.
"""

import sys
import os
import torch
import numpy as np
import soundfile as sf
import logging
from datetime import datetime
import time
import random
import nltk

nltk.download('punkt')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import EMO_MAP
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

# Global the tokenizer since we call it twice
tokenizer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "Spark-TTS-0.5B", "LLM"))
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

# Global cache for reuse
_cached_model_instance = None


# "Smart Text Split" - Builds segments based on sentences. Avoids mid-sentence chopping.
# Now works based on tokens.
def split_text_smart(text, threshold):

    sentences = sent_tokenize(text)
    segments = []
    current_segment = []
    current_token_count = 0

    for sentence in sentences:
        token_count = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_token_count + token_count > threshold:
            segments.append(' '.join(current_segment))
            current_segment = [sentence]
            current_token_count = token_count
        else:
            current_segment.append(sentence)
            current_token_count += token_count

    if current_segment:
        segments.append(' '.join(current_segment))

## DEBUG FOR SEGMENTATION ##
    for i, seg in enumerate(segments):
        print(f"\n===== Segment {i} =====")
        print(f"Words: {len(seg.split())}")
        print(f"Tokens: {len(tokenizer.encode(seg, add_special_tokens=False))}")
        print(seg)
        print("=" * 40)
    return segments


def generate_tts_audio(
    text,
    model_dir=None,
    device=None,
    prompt_speech_path=None,
    prompt_text=None,
    gender=None,
    pitch=None,
    speed=None,
    emotion=None,
    save_dir="example/results",
    segmentation_threshold=500,
    seed=None,
    model=None,
    skip_model_init=False,
    temperature=None,
    top_k=None,
    top_p=None,
):
    
    """
    Generates TTS audio from input text, splitting into segments if necessary.

    Args:
        text (str): Input text for speech synthesis.
        model_dir (str): Path to the model directory.
        device (str): Device identifier (e.g., "cuda:0" or "cpu"). - detected automatically now
        prompt_speech_path (str, optional): Path to prompt audio for cloning.
        prompt_text (str, optional): Transcript of prompt audio.
        gender (str, optional): Gender parameter ("male"/"female").
        pitch (str, optional): Pitch parameter (e.g., "moderate").
        speed (str, optional): Speed parameter (e.g., "moderate").
        emotion (str, optional): Emotion tag (e.g., "HAPPY", "SAD", "ANGRY").
        save_dir (str): Directory where generated audio will be saved.
        segmentation_threshold (int): Maximum number of tokens.
        seed (int, optional): Seed value for deterministic voice generation.

    Returns:
        str: The unique file path where the generated audio is saved.
    """
    # ============================== OPTIONS REFERENCE ==============================
    #  Gender options: "male", "female"
    #  Pitch options: "very_low", "low", "moderate", "high", "very_high"
    #  Speed options: same as pitch
    #  Emotion options: list from token_parser.py EMO_MAP keys
    #  Seed: any integer (e.g., 1337, 42, 123456) = same voice (mostly)
    # ==============================================================================

    if model_dir is None:
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "Spark-TTS-0.5B"))


# Detect CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

# Generates one random tensor int of 32-bit value from detected device.
# or one random number between 1 and 111 if you swap them
    if seed is None:
        seed = int(torch.randint(low=0, high=4294967295, size=(1, 1), device=device))
        #seed = int(torch.randint(low=1, high=111, size=(1, 1), device=device))


# Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda:0":
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Seed set to: {seed}")

# Save path with seed in filename
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    save_path = os.path.join(save_dir, f"{timestamp}_{seed}.wav")

    global _cached_model_instance

    if not skip_model_init or model is None:
        if _cached_model_instance is None:
            logging.info("Initializing TTS model...")
            if not prompt_speech_path:
                logging.info(f"Using Gender: {gender or 'default'}, Pitch: {pitch or 'default'}, Speed: {speed or 'default'}, Emotion: {emotion or 'none'}, Seed: {seed or 'random'}")
            model = SparkTTS(model_dir, torch.device(device))
            _cached_model_instance = model
        else:
            model = _cached_model_instance

# tokenize text, tokenize prompt text, add, see if more than 512
# if more than 512; subtract prompt_text_token_len from 512.
# set segments to split_text_smart(test, free space)

    tt_count = len(tokenizer.encode(text, add_special_tokens=False))
    pt_count = len(tokenizer.encode(prompt_text, add_special_tokens=False)) if prompt_text else 0
    t_total = tt_count + pt_count
    remainder = segmentation_threshold - pt_count
    logging.info(f"tt: {tt_count}, pt: {pt_count}, used: {t_total}, free: {remainder} total: {segmentation_threshold}")
    logging.info(f"temp: {temperature}, k: {top_k}, p: {top_p}, speed: {speed}")

# at some point all of that will be replaced with something like:
# remainder = segmentation_threshold - ((len(tokenizer.encode(text, add_special_tokens=False))) + (len(tokenizer.encode(prompt_text, add_special_tokens=False)) if prompt_text else 0))

# If you've only got 15 tokens free, you're gonna have a bad time. If you have less than 40, then you might be ok.
    if (remainder < 15):
        logging.critical("Fatal: Free Token Window Too Small ( < 15)")
        sys.exit(1)
    if (remainder < 40):
        logging.warning("Warning: Free Token Window < 40. Results may be bad.")

# Does segmentation if too big.
    if t_total > segmentation_threshold:
        logging.info("Text exceeds threshold; splitting into segments...")
        segments = split_text_smart(text, remainder)
        wavs = []
        for seg in segments:
            with torch.no_grad():
                wav = model.inference(
                    seg,
                    prompt_speech_path,
                    prompt_text=prompt_text,
                    gender=gender,
                    pitch=pitch,
                    speed=speed,
                    emotion=emotion,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            wavs.append(wav)
        final_wav = np.concatenate(wavs, axis=0)
    else:
        with torch.no_grad():
            final_wav = model.inference(
                text,
                prompt_speech_path,
                prompt_text=prompt_text,
                gender=gender,
                pitch=pitch,
                speed=speed,
                emotion=emotion,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

    sf.write(save_path, final_wav, samplerate=16000)
    logging.info(f"Audio saved at: {save_path}")
    return save_path


# Example CLI usage
if __name__ == "__main__":
    import argparse


    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_audio", type=str, help="Path to audio file for voice cloning")
    parser.add_argument("--prompt_text", type=str, help="Transcript text for the prompt audio (optional)")
    parser.add_argument("--text", type=str, help="Text to generate", required=False)
    parser.add_argument("--text_file", type=str, help="Path to .txt file with input text")
    parser.add_argument("--gender", type=str, choices=["male", "female"], default=None)
    parser.add_argument("--pitch", type=str, choices=["very_low", "low", "moderate", "high", "very_high"], default="moderate")
    parser.add_argument("--speed", type=str, choices=["very_low", "low", "moderate", "high", "very_high"], default="moderate")
    parser.add_argument("--emotion", type=str, choices=list(EMO_MAP.keys()), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--segmentation_threshold", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    # ---------------- Argument Validation Block ---------------- NEW! SPECIAL!!!EXTRA SPICY!!!
    if not args.prompt_audio and not args.gender:
        print("Error: You must provide either --gender (male/female) or --prompt_audio for voice cloning.")
        print("Example 1: python tts_cli.py --text \"Hello there.\" --gender female")
        print("Example 2: python tts_cli.py --text \"Hello there.\" --prompt_audio sample.wav")
        sys.exit(1)

    # --------------- Emotions ------------
    if args.emotion:
        logging.warning("Emotion input is experimental — model may not reflect emotion changes reliably or at all.")

    # There used to be a bunch of code around this guy but I replaced it all.
    # So now it stands alone making sure the variable is set.
#    segmentation_threshold = args.segmentation_threshold
#    temperature=args.temperature
#    top_k=args.top_k
#    top_p=args.top_p


    # Allow loading text from a file if provided
    if args.text_file:
        if os.path.exists(args.text_file):
            with open(args.text_file, "r", encoding="utf-8") as f:
                args.text = f.read().strip()
        else:
            raise FileNotFoundError(f"Text file not found: {args.text_file}")

    # If Not Provided Text or Text File
    if not args.text:
        raise ValueError("You must provide either --text or --text_file.")

    # Voice Cloning Mode Overrides
    if args.prompt_audio:
        # Normalize path + validate
        args.prompt_audio = os.path.abspath(args.prompt_audio)
        if not os.path.exists(args.prompt_audio):
            logging.error(f"Prompt audio file not found: {args.prompt_audio}")
            sys.exit(1)

        # Log cloning info
        logging.info("Voice cloning mode enabled")
        logging.info(f"Cloning from: {args.prompt_audio}")

        # Bonus: Log audio info
        try:
            info = sf.info(args.prompt_audio)
            logging.info(f"Prompt duration: {info.duration:.2f} seconds | Sample Rate: {info.samplerate}")
        except Exception as e:
            logging.warning(f"Could not read prompt audio info: {e}")

        # Override pitch/speed/gender
        if args.gender or args.pitch or args.speed:
            print("Warning: Voice cloning mode detected — ignoring gender/pitch/speed settings.")
        args.gender = None
        args.pitch = None
#        args.speed = None

    # Start timing
    start_time = time.time()

    output_file = generate_tts_audio(
        text=args.text,
        gender=args.gender,
        pitch=args.pitch,
        speed=args.speed,
        emotion=args.emotion,
        seed=args.seed,
        segmentation_threshold=args.segmentation_threshold,
        prompt_speech_path=args.prompt_audio,
        prompt_text=args.prompt_text,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # End timing
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Generated audio file: {output_file}")
    print(f"Generation time: {elapsed:.2f} seconds")


