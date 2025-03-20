from transformers import pipeline
import re
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
class PSYEVAL:
    def __init__(self, texts):
        self.texts = texts
        self.emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        self.save_path = 'e:/utilitiesBackUp/class/2409/CSC3160/A4(1)/A4_122090847/A4_122090847/src/saved_model'
        self.split_texts = self.__split_text()
    def HFT_eval(self):
        def hugging_face_transformer(texts):

            eng = ""
            chi = ""
            for char in texts:
                if ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
                    eng += char
                else:
                    chi += char
            if len(eng) / (len(chi) + 1) <= 0.01:
                classifier = pipeline("sentiment-analysis",model="uer/roberta-base-finetuned-jd-binary-chinese", device=0)
                texts = re.split(r'[，。！？/；]', texts)
            else:
                classifier = pipeline("sentiment-analysis",model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=0)
                texts = re.split(r'[,.!?/;]', texts)
            if texts[-1] == '': 
                texts = texts[:-1]

            results = classifier(texts)
            return results
        results = hugging_face_transformer(self.texts)
        return np.array([[result['score'] for _ in range(6)]for result in results])

    def __split_text(self):
       
        new_texts = []
        
        new_texts = re.split(r'[，。,.！!；;?？/]', self.texts)
        
        if new_texts[-1] == '': 
            new_texts = new_texts[:-1]
        return new_texts
    def __load_model(self):
        #print(f"Loading model and tokenizer from {self.save_path}...")
        tokenizer = BertTokenizer.from_pretrained(self.save_path)
        model = BertForSequenceClassification.from_pretrained(self.save_path)
        return model, tokenizer
    def Bert_eval(self):

        model, tokenizer = self.__load_model()
        def __emotion_evaluation(texts, model, tokenizer, emotion_labels):
            #print("Evaluating emotions with scores...")
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)  
            
            results = []
            for i, text in enumerate(texts):
                emotion_scores = {emotion_labels[j]: logits[i][j].item() for j in range(len(emotion_labels))}

                for emotion, score in emotion_scores.items():
                    if emotion == 'sadness' or emotion == 'fear' or emotion == 'anger':
                        emotion_scores[emotion] = -score
                predicted_emotion = emotion_labels[predictions[i]]
                results.append({"text": text, "predicted_emotion": predicted_emotion, "scores": emotion_scores})
            return results

        results = __emotion_evaluation(self.split_texts, model, tokenizer, self.emotion_labels)
        return np.array([
            [result['scores'][emotion] for emotion in self.emotion_labels] for result in results])

    def eval(self):
        matrix = np.dot(self.Bert_eval(), self.HFT_eval().T)
        #print(matrix.shape[0])
        return np.sum(matrix) / matrix.shape[0]