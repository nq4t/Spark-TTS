

## Spark-TTS

### Overview

This is a modified fork. For the official repository and original readme, please visit the [official Spark-TTS repository](https://github.com/SparkAudio/Spark-TTS)

This repository of Spark-TTS merges improvements by [ActePuKc](https://github.com/AcTePuKc) as originally mentioned in [this thread](https://github.com/SparkAudio/Spark-TTS/issues/10). It added text segmentation to prevent model overflow as well as better argument passing to inference.

I further refined upon this by making segementation both sentence-aware as well as work on the actual token count being sent to the model. I also exposed temperature, top_k, and top_p settings; as well as passed speed during voice cloning. I also created a shell script that aids in executing inference; and works well in modern text editors/ide with integrated terminal.

### Notes/Usage/Requirements

(An already working Spark-TTS environment.)[https://github.com/SparkAudio/Spark-TTS#install]

The segmentation uses nltk and and punkt. In your sparktts environment:

` pip install nltk && python -m nltk.downloader punkt `

### Command Line Interface

` python3 tts_cli.py --argument data --argument2 data `

Here's the list of arguments:

```
--prompt_audio            - .wav file of voice to be cloned
--prompt_text             - text of speaker from .wav file
--text                    - text to speak
--text_file               - text file to speak from
--gender                  - gender for voice creation
--pitch                   - pitch for voice creation
--speed                   - speed of generated text
--emotion                 - emotion for voice creation (experimental, doesn't really work)
--seed                    - number used to seed RNG
--segmentation_threshold  - the token count to use for segmentation
--temperature             - the temperature for generation
--top_k                   - top_k
--top_p                   - top_p
```

### Director Shell Script

The script does not take any arguments at this time. You have to edit the file before executing it. It was designed to be used in a modern text editor/ide with integrated shell; like VSCode, VScodium, Kate, etc. It takes most of the guess work out of building the argument string up for the interface. It also adds voice cloning presets, allowing you to just specify which voice you want to clone.

## ⚠️ Usage Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities.

- Ensure compliance with local laws and regulations when using this model and uphold ethical standards.

- The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.
