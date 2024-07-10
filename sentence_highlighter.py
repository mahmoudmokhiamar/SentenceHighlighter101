import os
import torch
import colorsys
import numpy as np
from nltk import sent_tokenize
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class sentenceHighlighter(object):
    def __init__(self,model=None,model_name="bert-base-uncased"):
        self.model = model 
        self.model_name = model_name
        self.sentences = []
        self.embeddings = []
        self.color_map = None
        self.similarity_matrix = None
    
    #read each file and tronsform the text into sentences by using sent_tokenize.
    def read_files(self,files):
        for file in files:
            with open(file, 'r') as f:
                text = f.read()
                self.sentences.extend(sent_tokenize(text))

    #transform the sentences into embeddings to be used by the model.
    def transform_sentences(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertModel.from_pretrained(self.model_name)
        for sentence in self.sentences:
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs)
            self.embeddings.append(outputs.pooler_output.detach().numpy())
        
        #to transform 3d array into 2d array
        self.embeddings = np.vstack(self.embeddings)

    #calculate the cosine similarity between the embeddings of the sentences.
    #the cosine similarity is used to determine the similarity between two vectors.
    def calculate_similarity(self):
        self.similarity_matrix = cosine_similarity(self.embeddings)
        self.color_map = {}
        colors = [colorsys.hsv_to_rgb(i/len(self.embeddings), 0.5, 0.9) for i in range(len(self.embeddings))]

        for i, row in enumerate(self.similarity_matrix):
            similar_indices = np.argsort(-row)[:2] #get the top 2 most similar sentences
            for idx in similar_indices:
                self.color_map[idx] = colors[i]

    #convert the rgb values to hex values. Since html uses hex values to represent colors.
    def rgb_to_hex(self,rgb):
        return '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
    
    ##############################################################################################################
    # def generate_html(self, output_file='highlighted_sentences2.html'):                                                  
    #     html_output = '<html><body>'
    #     for i, sentence in enumerate(self.sentences):
    #         color = self.rgb_to_hex(self.color_map.get(i, (1, 1, 1)))  # Default to white if not found
    #         html_output += f'<span style="background-color: {color}">{sentence}</span> '
    #     html_output += '</body></html>'
    #     html_output = html_output.replace('</span> ', '</span><br>')
    #     with open(output_file, 'w', encoding='utf-8') as file:
    #         file.write(html_output)
    #     print(f'HTML file generated: {output_file}')
    ##############################################################################################################

    #for the none-styled version comment this code and uncomment the above code.
    def generate_html(self, output_file='highlighted_sentences.html'):
        static_html = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Sentence Highlighter</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                }}
                .highlight {{
                    font-size: 1.5em;  /* Adjusted size for the highlight class */
                    color: #00ff00; /* bright green */
                }}
                h1 {{
                    text-align: center;
                }}
                .highlighted-sentences {{
                    font-family: monospace;
                    font-size: 18px;
                }}
            </style>
        </head>
        <body>
            <h1>Welcome to <span class="highlight">Sentence Highlighter!</span></h1>
            <div class="highlighted-sentences">
        '''
        
        dynamic_html = ''
        for i, sentence in enumerate(self.sentences):
            color = self.rgb_to_hex(self.color_map.get(i, (1, 1, 1)))  # Default to white if not found
            dynamic_html += f'<span style="background-color: {color}" style="text-align:left;">{sentence}</span><br>'

        # Close the HTML structure
        closing_html = '''
            </div>
        </body>
        </html>
        '''

        html_output = static_html + dynamic_html + closing_html

        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(html_output)

        print(f'HTML file generated: {output_file}')


sentenceHighlighter = sentenceHighlighter()
sentenceHighlighter.read_files(["file1.txt","file2.txt"])
sentenceHighlighter.transform_sentences()
sentenceHighlighter.calculate_similarity()
sentenceHighlighter.generate_html()

