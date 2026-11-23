# StudyMate LoRA: a small model that plans and tutors like StudyMate

The engine still draws everything. The fine-tuned model only reads lesson text and answers with
`{topic, title, visual, explain, takeaway}`, fast enough for a CPU.

## 1. Build the dataset (any PC, teacher model on a GPU is best)
```
python Engine\v4\lora\make_dataset.py uploads C:\path\to\course_pdfs --teacher-host http://<GPU-PC>:11434 --teacher qwen3:32b --synthetic 2
```
- `visual` comes from the engine's rules when they know the topic (always correct), otherwise from the teacher,
  and every example is test-drawn; bad ones are dropped.
- Resumable: run it again with more PDFs and it only does the new chunks.
- Aim for 1,000-3,000 examples. `data/stats.json` shows how many drawing types are covered.

## 2. Train (needs an NVIDIA GPU: Louis's PC, Colab T4, Kaggle, or rented)
```
pip install unsloth
python train_unsloth.py --data data --base unsloth/Qwen3-4B-Instruct-2507
```
1.7B trains in minutes and runs fastest on a CPU; 4B is the best balance; 8B if you'll have a GPU.

## 3. Use it
```
ollama create studymate-tutor -f studymate-tutor\Modelfile
python main_v4.py notes.pdf --model studymate-tutor
```
StudyMate notices the name `studymate-*` and switches to the tiny prompt the model was trained on.
