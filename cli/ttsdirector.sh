#!/bin/bash
# SparkTTS CLI "Director" - Jay Moore/NQ4T - TPTEL


# Master Segmentation Threshold Value
# This is now tokens. Default count is 512.
# I choose 500 to be safe.
segmentation_threshold=500

## Voice Generation Settings ##
pitch=moderate
gender=female
#emotion=EXCITED    #OPTIONAL: Comment out to skip.

# Mode: clone or gen ##
mode=clone

## Required Settings:
speed=moderate  # NOT IGNORED DURING CLONING ANYMORE
temperature=.8  # You now have temp settings.
top_k=50        # And top_k
top_p=.9        # And top_p
# and you're welcome!

## Options: Comment out to disable ##
use_prompt_text=thiscanbeanything   #OPTIONAL: Comment out to skip.
#seed=777                           #OPTIONAL: Comment out to skip.

## Voice Cloning ##

# Choosing a voice preset will over-ride the prompt_text and prompt_speech_path entered here.
voice=""

# These are ignored if using a voice preset.
prompt_text="Put speech text here. Ignored if using voice preset."
prompt_speech_path="/path/to/file.wav"


## Say This ##

saythis="Say this junk."

## Voice Presets ##
# This is a switch/case statement. It makes the voice presets work. Starts with a_name), contains commands, ends with double semi-colon. Copy and fill in blanks to add.
# Basically the same function as a bunch of if/then statements. It, itself, is wrapped in an if/then statement so it only exceutes during clone mode.

[[ "mode" == "clone" ]] && {
case $voice in
#   presetname)
#       prompt_text="text here"
#       prompt_speech_path="/path/to/file.wav"
#       ;;
#   anothername)
#       prompt_text="text here"
#       prompt_speech_path="/path/to/file.wav"
#       ;;
    *) # This is the default action for when no match is found (for any reason)
        [[ ! -f "$prompt_speech_path" ]] && echo "Voice Reference Not Found" && exit
        [[ -z "$prompt_speech_path" ]] && echo "Voice Reference Not Specified" && exit
        [[ -z "$prompt_text" ]] && echo "No prompt_text. Ignoring. Optional."
        ;;
esac
}


## Command building logic ##

# Builds arguments. It's literally just a bunch of if statements.

# These are constant and always passed
arguments+=("--segmentation_threshold" "$segmentation_threshold" "--text" "$saythis" "--temperature" "$temperature" "--top_k" "$top_k" "--top_p" "$top_p" "--speed" "$speed")

# Dyanamic arguments
[[ "$mode" == "gen" ]] && {
    arguments+=("--gender" "$gender" "--pitch" "$pitch")
    [[ -n "$emotion" ]] && arguments+=("--emotion" "$emotion")
}

[[ "$mode" == "clone" ]] && {
    arguments+=("--prompt_audio" "$prompt_speech_path")
    [[ -n "$use_prompt_text" ]] && arguments+=("--prompt_text" "$prompt_text")
}

[[ -n "$seed" ]] && arguments+=("--seed" "$seed")
[[ -z "$saythis" ]] && echo "Give me something to say!" && exit

## Run the thing. ##
python3 tts_cli.py "${arguments[@]}"
