import torch
import tqdm
import os.path
import json
from PIL import Image
import torch.nn.functional as F
from transformers import AutoProcessor, BlipForImageTextRetrieval
import argparse
import clip
from torch.nn.functional import normalize
from typing import Any, Optional, Tuple, Union
import logging
from finetune import utils
import numpy as np

os.environ['TOKENIZERS_PARALLELISM']='true'

parser = argparse.ArgumentParser()
parser.add_argument('--retriever', type=str, default='clip', choices=["clip", "blip"])
parser.add_argument('--queries-path', type=str, default='dialogues/VisDial_v1.0_queries_val.json')
parser.add_argument('--ft-model-path', type=str)
parser.add_argument('--cache-corpus', type=str)
parser.add_argument('--data-dir', type=str, default='visdial/corpus')
parser.add_argument('--output-dir', type=str, default='logs')
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--num-rounds', type=int, default=11)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--split', action='store_true', help="load dialog (caption) in split")

cfg = {'corpus_bs': 256,
       'queries_bs': 256,
       'num_workers': 8,
       'sep_token': ', ',  # Separation between dialog rounds
       'queries_path': None,
       'corpus_path': 'Protocol/Search_Space_val_50k.json',
       'device': 'cuda:0',  # 'cpu'
       }

args = parser.parse_args()
retriever = args.retriever
queries_path = args.queries_path
cfg['data_dir'] = args.data_dir
cfg['queries_path'] = queries_path
cfg['split'] = args.split
cfg['finetuned_model_path']=args.ft_model_path
cfg['cache_corpus']=args.cache_corpus
cfg['K']=args.K
cfg['queries_bs']=args.batch_size
device = "cuda" if torch.cuda.is_available() else "cpu"
 
if args.output_dir:
    utils.mkdir(args.output_dir)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, 'test.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

class BlipForRetrieval(BlipForImageTextRetrieval):
    def get_text_features(self,
                          input_ids: torch.LongTensor,
                          attention_mask: Optional[torch.LongTensor] = None,
                          return_dict: Optional[bool] = None,
                          ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

        text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

        return text_feat

    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]

        image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        return image_feat

class ImageEmbedder:
    def __init__(self, model, preprocessor):
        """ model projects image to vector, processor load and prepare image to the model"""
        self.model = model
        self.processor = preprocessor


class Corpus(torch.utils.data.Dataset):
    """ Dataset class for the corpus images (the 50k potential candidates)"""
    def __init__(self, data_dir, corpus_path, preprocessor):
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        self.corpus = [os.path.join(data_dir, path) for path in self.corpus]
        self.preprocessor = preprocessor
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}

    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path):
        """ For finding a target image fast"""
        return self.path2id[path]

    def __getitem__(self, i):
        if retriever == 'blip':
            image = self.preprocessor(self.corpus[i])['pixel_values'][0]
        else:
            image = self.preprocessor(self.corpus[i])  # Load and prepare image
        return {'id': i, 'image': image}


class Queries(torch.utils.data.Dataset):
    """ Dataset class for the queries and their targets (dialog and image)"""
    def __init__(self, cfg, queries_path, txt_processors):
        with open(queries_path) as f:
            self.queries = json.load(f)

        self.dialog_length = None  # Set the dialog length to evaluate on
        self.cfg = cfg
        self.txt_processors = txt_processors

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        assert self.dialog_length is not None, "Please set self.dialog_length=<DIALOG_LENGTH> to any number [0,..,10]"
        target_path = os.path.join(self.cfg['data_dir'], self.queries[i]['img'])
        # Concatenate the partial dialog information with a predefined seperator.
        if self.cfg['split']:
            text = self.queries[i]['dialog'][self.dialog_length]
        else:
            text = self.cfg['sep_token'].join(self.queries[i]['dialog'][:self.dialog_length + 1])
        return {'text': text, 'target_path': target_path}


class PlugIREval:
    """ This class run the main evaluation process.
    """
    def __init__(self, cfg, dialog_encoder, image_embedder: ImageEmbedder, txt_processors):
        self.dialog_encoder = dialog_encoder  # In paper was referred as "Image Retriever"
        self.image_embedder = image_embedder  # Image encoder
        self.txt_processors = txt_processors

        self.cfg = cfg
        self.corpus = None
        self.corpus_dataset = Corpus(self.cfg['data_dir'], self.cfg['corpus_path'], self.image_embedder.processor)
        self.scores = {}
        self.ranks = []
        self.targets =[]

    def _get_recalls(self, dataloader, dialog_length):
        # Set dialog length
        dataloader.dataset.dialog_length = dialog_length
        recalls = []
        ranks = []
        targets = []
        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            target_ids = torch.tensor([self.corpus_dataset.path_to_index(p) for p in batch['target_path']]).unsqueeze(1).to(self.cfg['device'])
            # batch recalls
            pred_vec = F.normalize(self.dialog_encoder(batch['text']), dim=-1) # Nxd
            self.scores[i] = pred_vec @ self.corpus[1].T
            arg_ranks = torch.argsort(self.scores[i], descending=True, dim=1).long()
            target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
            recalls.append(target_recall)
            ranks.append(arg_ranks)
            targets.append(target_ids)
            self.scores[i] = self.scores[i].cpu()

        return torch.cat(recalls).cpu(), torch.cat(ranks).cpu(), torch.cat(targets).cpu()

    def run(self, hits_at):
        assert self.corpus, f"Prepare corpus first (self.index_corpus())"
        dataset = Queries(cfg, self.cfg['queries_path'], self.txt_processors)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.cfg['queries_bs'],
                                                 shuffle=False,
                                                 num_workers=self.cfg['num_workers'],
                                                 pin_memory=True,
                                                 drop_last=False
                                                 )
        hits_results = []
        ranks_results = []
        targets_results = []
        min_ranks = []
        for dl in range(args.num_rounds):
            logging.info(f"Calculate recalls for each dialogues of length {dl}...")
            dialog_recalls, ranks, targets = self._get_recalls(dataloader, dialog_length=dl)
            if dl == 0:
                min_ranks.append(dialog_recalls)
            else:
                min_ranks.append(torch.minimum(min_ranks[dl-1], dialog_recalls))
            hits_results.append(dialog_recalls)
            ranks_results.append(ranks)
            targets_results.append(targets)

        hits_results, temp_hits_results = cumulative_hits_per_round(
            torch.cat(hits_results),
            torch.cat(ranks_results),
            torch.cat(targets_results),
            hitting_recall=cfg['K'])
        hits_results = hits_results.tolist()
        temp_hits_results = temp_hits_results.tolist()
        logging.info(f"====== Results for Hits@{cfg['K']} ====== ")
        for dl in range(args.num_rounds):
            logging.info(f"\t Dialog Length: {dl}: {round(hits_results[dl], 2)}%")
        logging.info(f"====== Results for Recall@{cfg['K']} ====== ")
        for dl in range(args.num_rounds):
            logging.info(f"\t Dialog Length: {dl}: {round(temp_hits_results[dl], 2)}%")
        logging.info(f"====== Best log Rank Integral ====== ")
        bri = 0
        mean_best_rank_over_rounds = []
        std_best_rank_over_rounds = []
        for dl in range(args.num_rounds-1):
            bri += ((torch.log(min_ranks[dl]+1.) + torch.log(min_ranks[dl+1]+1.)) / 2).mean()
        bri /= args.num_rounds-1
        logging.info(f"\t BRI: {bri}")

    def index_corpus(self):
        """ Prepare corpus (image search space)"""
        # self.corpus = torch.arange(50000).to(cfg['device']), torch.randn(50_000, 512).to(cfg['device']).half()
        print(self.cfg['cache_corpus'])
        print(os.path.exists(self.cfg['cache_corpus']))
        if self.cfg['cache_corpus'] and os.path.exists(self.cfg['cache_corpus']):
            logging.info(f"<<<<Cached corpus has been loaded: {self.cfg['cache_corpus']} >>>>>")
            logging.info(f"Warning: Make sure this corpus has been indexed with the right image embedder!")
            self.corpus = torch.load(self.cfg['cache_corpus'])
            return
        # return
        dataloader = torch.utils.data.DataLoader(self.corpus_dataset,
                                                 batch_size=self.cfg['corpus_bs'],
                                                 shuffle=False,
                                                 num_workers=self.cfg['num_workers'],
                                                 pin_memory=True,
                                                 drop_last=False
                                                 )
        logging.info("Preparing corpus (search space)...")
        corpus_vectors = []
        corpus_ids = []
        for batch in tqdm.tqdm(dataloader):
            batch_vectors = F.normalize(self.image_embedder.model(batch['image'].to(self.cfg['device'])), dim=-1)
            corpus_vectors.append(batch_vectors)
            corpus_ids.append(batch['id'].to(self.cfg['device']))

        corpus_vectors = torch.cat(corpus_vectors)
        corpus_ids = torch.cat(corpus_ids)

        # sort by id: important!
        arg_ids = torch.argsort(corpus_ids)
        corpus_vectors = corpus_vectors[arg_ids]
        corpus_ids = corpus_ids[arg_ids]

        self.corpus = corpus_ids, corpus_vectors
        if self.cfg['cache_corpus']:
            torch.save(self.corpus, self.cfg['cache_corpus'])


def get_first_hitting_time(target_recall, hitting_recall=10):
    """ returns (11, n) tensor with hitting time in each round (0, 11). inf indicate a miss (no hit after 11 rounds) """
    target_recalls = target_recall.view(args.num_rounds, -1).T
    hits = (target_recalls < hitting_recall)

    final_hits = torch.inf * torch.ones(target_recalls.shape[0])

    hitting_times = []
    temp_hitting_times = []
    for ro_i in range(args.num_rounds):
        temp_hits = torch.inf * torch.ones(target_recalls.shape[0])
        rh = hits[:, ro_i]
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape) * ro_i)
        temp_hits[rh] = torch.min(temp_hits[rh], torch.ones(temp_hits[rh].shape) * ro_i)
        hitting_times.append(final_hits.clone())
        temp_hitting_times.append(temp_hits)

    return torch.stack(hitting_times), torch.stack(temp_hitting_times)


def cumulative_hits_per_round(target_recall, ranks, targets, hitting_recall=10):
    """ return calculation of avg number of hits until round x"""
    if type(hitting_recall) is tuple:
        assert len(hitting_recall) == 1
        hitting_recall = hitting_recall[0]

    ht_times, temp_ht_times = get_first_hitting_time(target_recall, hitting_recall)

    return ((ht_times < torch.inf).sum(dim=-1) * 100 / ht_times[0].shape[0]), ((temp_ht_times < torch.inf).sum(dim=-1) * 100 / temp_ht_times[0].shape[0])


def CLIP_ZERO_SHOT_BASELINE():
    # Install CLIP library from https://github.com/openai/CLIP
    # model, preprocess = clip.load("ViT-B/32", device='cpu')
    model, preprocess = clip.load("ViT-B/16", device='cpu')
    model = model.to(device)
    image_embedder = ImageEmbedder(lambda img: model.encode_image(img), lambda path: preprocess(Image.open(path)))
    # Note that CLIP supports only 77 tokens!! this is just a baseline.
    dialog_encoder = lambda text: model.encode_text(clip.tokenize(text, truncate=True).to(device))

    return dialog_encoder, image_embedder


def BLIP_ZERO_SHOT_BASELINE(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")

    if cfg['finetuned_model_path']:
        ckpt = torch.load(cfg['finetuned_model_path'], map_location="cpu")
        state_dict = ckpt['model']
        msg = model.load_state_dict(state_dict, strict=False)
        logging.info(f"Load pretrained model with msg: {msg}")

    model = model.to(device)

    image_embedder = ImageEmbedder(lambda img: model.get_image_features(img),
                                   lambda path: processor(images=Image.open(path), return_tensors='pt'))
    # if blip, Copus.getitem is image = self.preprocessor(self.corpus[i])['pixel_values']
    dialog_encoder = lambda text: model.get_text_features(**processor(text=text,
                                                                      padding=True,
                                                                      truncation=True,
                                                                      return_tensors="pt"))
    return dialog_encoder, image_embedder

with torch.no_grad():
    txt_processors = None
    if retriever == 'blip':
        dialog_encoder, image_embedder = BLIP_ZERO_SHOT_BASELINE(cfg)
    else:
        dialog_encoder, image_embedder = CLIP_ZERO_SHOT_BASELINE()
    evaluator = PlugIREval(cfg, dialog_encoder, image_embedder, txt_processors)
    evaluator.index_corpus()
    evaluator.run(hits_at=cfg['K'])  # Hit@10 as in the paper

