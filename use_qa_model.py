import tkinter as tk
from tkinter import scrolledtext, messagebox
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class QAApplication:
    def __init__(self, master):
        self.master = master
        master.title("Question Answering System")
        master.geometry("600x400")

        self.load_model_and_data()
        self.create_widgets()

    def load_model_and_data(self):
        model_path = "./qa_model"  # Adjust this path as needed
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)

        # Load documents
        df = pd.read_csv('news_data.csv')  # Adjust this path as needed
        df['text'] = df['title'] + " " + df['description']
        self.documents = df['text'].tolist()

        # Prepare TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    def create_widgets(self):
        # Question input
        tk.Label(self.master, text="Enter your question:").pack(pady=10)
        self.question_entry = tk.Entry(self.master, width=50)
        self.question_entry.pack(pady=5)
        
        # Submit button
        tk.Button(self.master, text="Ask", command=self.answer_question).pack(pady=10)

        # Answer display
        tk.Label(self.master, text="Answer:").pack(pady=5)
        self.answer_text = scrolledtext.ScrolledText(self.master, width=70, height=10)
        self.answer_text.pack(pady=5)

    def retrieve_relevant_document(self, question, top_k=1):
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.doc_vectors)
        top_indices = similarities.argsort()[0][-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def get_answer(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Find the tokens with the highest start and end scores
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        # Ensure end_index is not before start_index
        if end_index < start_index:
            end_index = start_index

        # Convert token indices to character indices
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        answer = self.tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])

        # Clean up the answer
        answer = answer.strip()
        answer = answer.replace('[CLS]', '').replace('[SEP]', '').strip()

        # If answer is empty or just punctuation, return a portion of the context
        if not answer or not any(c.isalnum() for c in answer):
            return context[:100] + "..."  # Return first 100 characters of context

        return answer

    def answer_question(self):
        question = self.question_entry.get()
        if not question:
            messagebox.showwarning("Warning", "Please enter a question.")
            return

        relevant_docs = self.retrieve_relevant_document(question, top_k=3)  # Retrieve top 3 documents
        if relevant_docs:
            context = " ".join(relevant_docs)[:1000]  # Combine contexts, limit to 1000 characters
            answer = self.get_answer(question, context)
            
            self.answer_text.delete('1.0', tk.END)
            self.answer_text.insert(tk.END, f"Context: {context[:200]}...\n\n")
            self.answer_text.insert(tk.END, f"Answer: {answer}")
        else:
            self.answer_text.delete('1.0', tk.END)
            self.answer_text.insert(tk.END, "No relevant context found for the question.")

def main():
    root = tk.Tk()
    app = QAApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()