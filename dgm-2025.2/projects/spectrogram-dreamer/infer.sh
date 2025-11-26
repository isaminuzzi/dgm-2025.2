# On our way to San Francisco, we were listening to reggae music and smoking weed.
python infer.py \
  --model checkpoints/dreamer_20251125_050735/best_model.pt \
  --audio data/1_validated-audio/common_voice_en_10047.mp3 \
  --styles data/3_style-vectors/common_voice_en_100039 \
  --output result.wav \


# The best view comes after the worst climb
# python inference_with_style.py \
#   --model mlruns/789017296987570404/0d2a4e3ed58a4a3da945cc9cb861c675/artifacts/best_model.pt \
#   --input example_audio.mp3 \
#   --mode transfer \
#   --style_a data/3_style-vectors/common_voice_en_10047/common_voice_en_10047_0000.pt \
#   --temp 0.5

