import sys, os
from dotenv import load_dotenv
from langchain_community.embeddings import SambaStudioEmbeddings
from langchain_community.llms.sambanova import SambaStudio
from keybert import KeyBERT, KeyLLM
from keybert.llm import TextGeneration
from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer
from src.custom_models import CustomEmbedder, CustomTextGeneration
from src.keyLLM import CustomKeyLLM
from src.utils import read_txt_files, extract_first_values
import torch
import pickle
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
# repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(current_dir)
load_dotenv(os.path.join(kit_dir, '.env'))
from vectordb.vector_db import VectorDb

class KeywordExtractor:
    """
    Extract keywords using KeyBert
    """
    def __init__(self, docs, use_bert=True, use_llm=False) -> None:
        self.docs = docs
        self.use_bert = use_bert
        self.use_llm = use_llm
        self.load_models()
        self.create_kw_models()
        
    def load_models(self, coe=False, **model_param) -> None:
        # 1. create vector db
        vdb = VectorDb()
        # 2. load embedding model
        self.embedding_model = vdb.load_embedding_model(type = "sambastudio", coe = coe, **model_param)

        # 3. load llm model
        if self.use_llm:
            self.llm = SambaStudio(
                streaming=False,
                model_kwargs={
                    'max_tokens_to_generate': 512,
                    'select_expert': 'Mistral-7B-Instruct-v0.2', #'Meta-Llama-3-8B-Instruct',
                    'temperature': 0.0,
                    'do_sample': False,
                },
            )

    def create_kw_models(self) -> None:
        # 4. create kw_model
        self.custom_embedder = CustomEmbedder(embedding_model=self.embedding_model)
        # pass custom backend to keybert
        self.kw_bert_model = KeyBERT(model=self.custom_embedder)
        # load it in KeyLLM
        if self.use_bert and self.use_llm:
            raise NotImplementedError("Not support both bert and generative llm yet.")
        elif self.use_bert:
            self.kw_llm_model = CustomKeyLLM(self.kw_bert_model)
        elif self.use_llm:
            self.text_generator = CustomTextGeneration(self.llm)
            self.kw_llm_model = KeyLLM(self.text_generator)

    def docs_embedding(self) -> None:
        # embed docs
        self.docs_embed = self.custom_embedder.embed(documents=self.docs)

    def extract_keywords(self, 
                         use_clusters=True, 
                         use_keyphrase=True, 
                         keyphrase_ngram_range: tuple[int, int] = (1,1)) -> set:
        # retrieve keywords
        vectorizer = None
        if use_keyphrase:
            vectorizer = KeyphraseTfidfVectorizer()
            keyphrase_ngram_range = None

        if use_clusters and self.use_bert:
            _, keywords = self.kw_llm_model.extract_keywords(docs=self.docs, embeddings=torch.as_tensor(self.docs_embed), threshold=.9,use_maxsum=True,nr_candidates=20, top_n=5, vectorizer=vectorizer, keyphrase_ngram_range=keyphrase_ngram_range)
            self.keywords = extract_first_values(keywords, return_list=False)
        elif use_clusters and self.use_llm:
            _, self.keywords = self.kw_llm_model.extract_keywords(docs=self.docs, embeddings=torch.as_tensor(self.docs_embed), threshold=.9)
        elif not use_clusters and self.use_bert:
            keywords = self.kw_bert_model.extract_keywords(docs[:1], keyphrase_ngram_range=keyphrase_ngram_range, use_maxsum=True,nr_candidates=20, top_n=5, vectorizer=vectorizer)
            self.keywords = extract_first_values(keywords, return_list=False)
        elif not use_clusters and self.use_llm:
            self.keywords = self.text_generator.extract_keywords(docs=self.docs, threshold=.9)
        # for i in range(len(keywords)):
        #     print(keywords[i])
        return self.keywords

    def save_keywords(self, save_filepath="./keywords/keywords_sambatune.pkl") -> None:
        # save keywords
        with open(save_filepath, "wb") as file:
            pickle.dump(self.keywords, file)

if __name__ == "__main__":
    # load docs
    docs_filepath = current_dir + "/context_articles"
    file_folder = docs_filepath 
    docs = read_txt_files(file_folder)
    # extract keywords
    kw_etr = KeywordExtractor(docs, use_bert=True)
    kw_etr.docs_embedding()
    keywords = kw_etr.extract_keywords()
    kw_etr.save_keywords()