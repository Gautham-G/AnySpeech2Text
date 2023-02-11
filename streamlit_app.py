# Imports
import streamlit as st
from transformers import pipeline
from PIL import Image
import os
import whisper
from tempfile import NamedTemporaryFile

# Pipeline - pretrained model
model = whisper.load_model("small")

# Display
st.title("Transcribe text")
file_name = st.file_uploader("Upload Audio File (<10 sec please!)", type=[".wav"])

# Audio inference       
def inference(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    _, probs = model.detect_language(mel)
    
    options = whisper.DecodingOptions(fp16 = False, task = "translate")
    result = whisper.decode(model, mel, options)
    
    print(result.text)
    return result.text


# Streamlit
if file_name is not None:
    with NamedTemporaryFile(suffix="mp3") as temp:
        temp.write(file_name.getvalue())
        temp.seek(0)

        col1, col2 = st.columns(2)

        
        col1.text("..Your translation is loading!")
        transcription = inference(temp.name)

        st.text(transcription)
