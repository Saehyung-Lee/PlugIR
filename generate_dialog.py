import torch
import openai
from tqdm import tqdm
import time
import json
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, BlipForImageTextRetrieval
from torch.nn.functional import normalize
from typing import Optional
import argparse
from fast_pytorch_kmeans import KMeans


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


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1021)
parser.add_argument('--s_idx', type=int, default=0)
parser.add_argument('--e_idx', type=int, default=2064)
parser.add_argument('--api_key', type=str, default=None)
parser.add_argument('--q_n', type=int, default=1)
parser.add_argument('--recall_hitting', type=int, default=10)
parser.add_argument('--thres_low', type=int, default=500)
parser.add_argument('--reconstruct', action='store_true')
parser.add_argument('--referring', action='store_true')
parser.add_argument('--filtering', action='store_true')
parser.add_argument('--select', action='store_true')
args = parser.parse_args()

SEED = args.seed
s_idx = args.s_idx
e_idx = args.e_idx
api_key = args.api_key
q_n = args.q_n
recall_hitting = args.recall_hitting
threshold_low = args.thres_low
reconstruct = args.reconstruct
referring = args.referring
filtering = args.filtering
select = args.select

openai.api_key = api_key
model_id = "Salesforce/blip2-flan-t5-xl"
device = "cuda:0"

processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
blip = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)

processor2 = Blip2Processor.from_pretrained(model_id)
blip2 = Blip2ForConditionalGeneration.from_pretrained(model_id,
                                                      device_map={"": 0},
                                                      torch_dtype=torch.float16)

with open('./ChatIR/dialogues/VisDial_v1.0_queries_val.json', 'r') as diag_json:
    visdial = json.load(diag_json)
with open('./ChatIR/ChatIR_Protocol/Search_Space_val_50k.json', 'r') as ss_json:
    search_space = json.load(ss_json)
with open('./ChatIR/ChatIR_Protocol/visdial_captions.json', 'r') as ss_cap_json:
    captions = json.load(ss_cap_json)

img_embs = torch.load('./ChatIR/temp/corpus_blip_large.pth', map_location=device)[1]
kmeans = KMeans(n_clusters=10, mode='cosine', verbose=0)


def reconstruct_dialog(dial, temp=.0, model='gpt-3.5-turbo-0613'):
    caption = dial[0]
    dialog = ', '.join(dial[1:])

    retry_count = 0
    dialog_examplar = ', '.join(["is this in a park? yes, i believe it is", "are there others around? no, she is alone",
                                 "does she have a collection bucket? no", "is her hair long? yes, pretty long",
                                 "is she wearing a dress? i don't think so, hard to tell",
                                 "does she have shoes on? yes, flip flops", "is there grass nearby? yes, everywhere",
                                 "is it a sunny day? yes", "are there trees? in the background there are trees",
                                 "is the guitar new? i don't think so"])
    messages = [{"role": "system",
                 "content": "Your role is to reconstruct the [Caption] with the additional information given by following [Dialogue]. The reconstructed [New Caption] should be concise and in appropriate form to retrieve a target image from a pool of candidate images"}]
    messages.append({"role": "user",
                     "content": f"[Caption]: a woman sits on a bench holding a guitar in her lap [Dialogue]: {dialog_examplar}  [New Caption]: "})
    messages.append({"role": "assistant",
                     "content": "a woman with pretty long hair sits alone on a grassy bench in a park on a sunny day, holding a guitar in her lap without a collection bucket, wearing flip flops, with trees in the background, with a slightly worn guitar"})
    messages.append({"role": "user", "content": f"[Caption]: {caption} [Dialogue]: {dialog}  [New Caption]: "})
    while True:
        try:
            response = openai.ChatCompletion.create(model=model,
                                                    messages=messages,
                                                    n=1,
                                                    temperature=temp,
                                                    request_timeout=10,
                                                    seed=SEED,
                                                    max_tokens=512)
            break
        except Exception as e:
            print(e)
            time.sleep(3)
            retry_count += 1
            if retry_count > 5:
                print("Too many retrials for generation")
                return f"Error: {e}"
            continue
    recon = response['choices'][0]['message']['content']

    return recon


def get_text_features(text):
    features = blip.get_text_features(**processor(text=text, padding=True, return_tensors="pt"))

    return features


def search_imgs(query="", img_embs=None, search_space=None, k=10):
    query_emb = get_text_features(query)
    query_emb_norm = normalize(query_emb, dim=-1)
    cos_sim = torch.matmul(query_emb_norm, img_embs.T).squeeze()
    related_indices = cos_sim.sort()[1][-k:]
    related_imgs = []
    for idx in range(k):
        related_imgs.append(search_space[related_indices[idx].item()])

    return related_imgs, related_indices, cos_sim


def get_related_captions(caption_recon, round=1):
    caps = []
    # related_size = 100 - (round-1)*10
    related_size = int(threshold_low - (round-1) * (threshold_low / 10.0))
    emb = normalize(get_text_features(caption_recon), dim=-1)
    sim = torch.matmul(emb, img_embs.T).squeeze()
    topk = sim.argsort()[-related_size:]
    img_embs_topk = img_embs[topk]

    entropies = torch.zeros([related_size])
    for i in range(related_size):
        cap = captions[topk[i].item()]['caption']
        emb = normalize(get_text_features(cap), dim=-1)
        sim = torch.matmul(emb, img_embs_topk.T).squeeze()
        p = torch.nn.functional.softmax(sim, dim=0)
        entropy = (-p * p.log()).sum().detach().cpu()
        entropies[i] += entropy
    idx_entropies_sorted = entropies.argsort()

    cluster_label = kmeans.fit_predict(img_embs_topk)
    cluster_label_sorted = cluster_label[idx_entropies_sorted]
    for i in range(10):
        if (cluster_label_sorted == i).any():
            idx_c = (cluster_label_sorted == i).nonzero().squeeze().min()
            caps.append(captions[topk[idx_entropies_sorted[idx_c]].item()]['caption'][0])

    # for i in range(10):
    #     caps.append(captions[topk[entropies.argsort()[i]].item()]['caption'][0])

    return caps


def get_referring_prompt(caption="", img_embs=None, k=10, round=1, search_space=None):
    img_paths, top_k, cos_sims = search_imgs(caption, img_embs, search_space, k=k)
    fakes = get_related_captions(caption, round)

    prompt_sys = ""
    prompt_sys += "You should leverage the 'Fake Information' that is related to the target image "
    prompt_sys += "corresponding to the caption but does not match the target image."

    prompt_fake = ""

    for i in range(len(fakes)):
        prompt_fake += str(i) + '. ' + fakes[i] + '\n'

    return prompt_sys, prompt_fake, top_k, cos_sims


def generate_questions(descrip, n=0, model='gpt-3.5-turbo'):
    prompt_sys = ""
    prompt_example = ""
    prompt_assi = ""
    prompt_user = ""

    prompt_sys += "You are a proficient question generator tasked with aiding in the retrieval of a target image. "
    prompt_sys += "Your role is to generate questions about the target image of the description via "
    prompt_sys += "leveraging two key information sources:\n\n"
    prompt_sys += "[Description]: This is a concise explanation of the target image.\n"
    prompt_sys += "[Dialogue]: Comprising question and answer pairs that seek additional "
    prompt_sys += "details about the target image.\n"
    prompt_sys += "Your generated question about the description must be clear, succinct, and concise, "
    prompt_sys += "while differing from prior questions in the [Dialogue]."

    prompt_example += "[Description]\n"
    prompt_example += "a man is doing a trick on a skateboard\n"
    prompt_example += "\n[Dialogue]\n"
    prompt_example += "Question: What type of trick is the man performing on the skateboard?\n"
    prompt_example += "Answer: a jump\n"
    prompt_example += "Question: What is the location of the jump trick being performed?\n"
    prompt_example += "Answer: a skate park\n"
    prompt_example += "Question: "

    prompt_assi += "what is the outfit of the man performing the jump trick at a skate park?"

    prompt_user += "\n[Description]\n"
    prompt_user += descrip[0] + '\n'
    prompt_user += "\n[Dialogue]\n"

    for i in range(len(descrip) - 1):
        qa = descrip[i + 1]
        q = qa.split('? ')[0] + '?'
        a = qa.split('? ')[1]
        prompt_user += "Question: " + q + '\n'
        prompt_user += "Answer: " + a + '\n'

    prompt_user += "Question: "

    retry_count = 0

    messages = [{"role": "system", "content": prompt_sys}]
    messages.append({"role": "user", "content": prompt_example})
    messages.append({"role": "assistant", "content": prompt_assi})
    messages.append({"role": "user", "content": prompt_user})

    while True:
        try:
            response = openai.ChatCompletion.create(model=model,
                                                    messages=messages,
                                                    max_tokens=32,
                                                    temperature=0.5,
                                                    n=n,
                                                    request_timeout=10)
            break

        except Exception as e:
            print(e)
            time.sleep(3 * (retry_count + 1))
            retry_count += 1
            if retry_count > 5:
                print("Too many retrials for generation")

            continue

    # prompt_total = prompt_sys + '\n' + prompt_example + prompt_assi + '\n' + prompt_user

    return response


def generate_questions_referring(descrip, prompt_fake="", n=10, ques_prior=[], model='gpt-3.5-turbo'):
    prompt_sys = ""
    prompt_example = ""
    prompt_assi = ""
    prompt_user = ""

    prompt_sys += "You are a proficient question generator tasked with aiding in the retrieval of a target image. "
    prompt_sys += "Your role is to generate questions about the target image of the description via "
    prompt_sys += "leveraging three key information sources:\n\n"
    prompt_sys += "[Retrieval Candidates]: These are captions of images which are the candidates "
    prompt_sys += "of the retrieval task for the target image described in [Description].\n"
    prompt_sys += "[Description]: This is a concise explanation of the target image.\n"
    prompt_sys += "[Dialogue]: Comprising question and answer pairs that seek additional "
    prompt_sys += "details about the target image.\n\n"
    prompt_sys += "You should craft a question that narrows down the options for "
    prompt_sys += "the attributes of the target image through "
    prompt_sys += "drawing the information from the retrieval candidates. "
    prompt_sys += "The generated question about the target image must be clear, succinct, and concise. "
    prompt_sys += "Also, the question should only be asked about common objects in the description and candidates, "
    prompt_sys += "which cannot be answered only from the description and the dialogue. "
    prompt_sys += "Please explain how did you utilize the information sources for generating a question.\n"

    prompt_example += "[Retrieval Candidates]\n"
    prompt_example += "0. man in yellow shirt\n"
    prompt_example += "1. a boy in a skateboard park\n"
    prompt_example += "2. the biker is performing a trick\n"
    prompt_example += "3. a man in a green hat doing half-pipe with a skateboard\n"
    prompt_example += "4. a skateboarding man catches the air in the midst of a trick\n"
    prompt_example += "[Description]\n"
    prompt_example += "a man is doing a trick on a skateboard\n"
    prompt_example += "[Dialogue]\n"
    prompt_example += "Question: what type of trick is the man performing on the skateboard?\n"
    prompt_example += "Answer: a jump\n"
    prompt_example += "Question: what is the location of the jump trick being performed?\n"
    prompt_example += "Answer: a skate park\n"
    prompt_example += "Question: "

    prompt_assi += "what is the outfit of the man performing the jump trick at a skate park?\n"
    prompt_assi += "Explanation:To generate a question about the description, I will utilize the "
    prompt_assi += "retrieval candidates that mention the outfit of the man. Candidates 0 and 3 "
    prompt_assi += "provide information about the man's wearing. "
    prompt_assi += "The description mentions the man's trick on a skateboard, and the dialogue mentions "
    prompt_assi += "the type and the location of the trick. "
    prompt_assi += "Since the attribute about the outfit is not appeared at the description and the dialogue, "
    prompt_assi += "the generated question cannot be answered from the information of "
    prompt_assi += "the description and the dialogue about the target image. "
    prompt_assi += "Also, the generated question is asking for the common objective, man, in the"
    prompt_assi += " descriptions and candidates, "
    prompt_assi += "not for the different objective from the description and the retrieval candidates 0 and 3, "
    prompt_assi += "for example, a shirt and a half-pipe."

    prompt_user += "[Retrieval Candidates]\n"
    prompt_user += prompt_fake

    prompt_user += "[Description]\n"
    prompt_user += descrip[0]
    prompt_user += "[Dialogue]\n"

    for i in range(len(descrip) - 1):
        qa = descrip[i + 1]
        q = qa.split('? ')[0] + '?'
        a = qa.split('? ')[1]
        prompt_user += "Question: " + q + '\n'
        prompt_user += "Answer: " + a + '\n'

    for i in range(len(ques_prior)):
        prompt_user += "Question: " + ques_prior[i] + '\n'
        prompt_user += "Answer:\n"

    prompt_user += "Question: "

    retry_count = 0

    messages = [{"role": "system", "content": prompt_sys}]
    messages.append({"role": "user", "content": prompt_example})
    messages.append({"role": "assistant", "content": prompt_assi})
    messages.append({"role": "user", "content": prompt_user})

    while True:
        try:
            response = openai.ChatCompletion.create(model=model,
                                                    messages=messages,
                                                    max_tokens=32,
                                                    request_timeout=10,
                                                    n=1,
                                                    temperature=0.5)
            break
        except Exception as e:
            print(e)
            time.sleep(3 * (retry_count + 1))
            retry_count += 1
            if retry_count > 5:
                print("Too many retrials for generation")
            continue

    # prompt_total = prompt_sys + '\n\n' + prompt_example + prompt_assi + '\n\n' + prompt_user

    return response


def filter_questions(context, question, model='gpt-3.5-turbo-0613'):
    prompt_sys = ""
    prompt_sys += "Answer the question only according to the given context. "
    prompt_sys += "If you cannot determine the answer or there are no objects "
    prompt_sys += "that are asked by the question in the context , answer \"Uncertain\"."


    messages = [{"role": "system", "content": prompt_sys}]

    prompt_user = ""
    prompt_user += "[Context]\n"
    prompt_user += context
    prompt_user += "\n[Question]\n"
    prompt_user += question
    prompt_user += "\n[Answer]\n"

    messages.append({"role": "user", "content": prompt_user})

    retry_count = 0

    while True:
        try:
            response = openai.ChatCompletion.create(model=model,
                                                    messages=messages,
                                                    max_tokens=10,
                                                    request_timeout=10,
                                                    n=1,
                                                    temperature=.0)
            break
        except Exception as e:
            print(e)
            time.sleep(3 * (retry_count + 1))
            retry_count += 1
            if retry_count > 5:
                print("Too many retrials for generation")
            continue

    ans = response['choices'][0]['message']['content']

    return ans.lower()


def select_question(caption_recon="", questions=[], cossim_prev=None,
                     k=10, img_embs=None, threshold=500, round=1):
    threshold = int(threshold - (round-1) * (threshold / 10.0))
    idx_related = cossim_prev.argsort()[-threshold:-k]
    p_prev = torch.nn.functional.softmax(cossim_prev[idx_related], dim=0)
    kl_divs = torch.zeros([len(questions)])

    for i, ques in enumerate(questions):
        caption_tmp = caption_recon + ", " + ques

        query_emb_tmp = normalize(get_text_features(caption_tmp), dim=-1)
        cossim_tmp = torch.matmul(query_emb_tmp, img_embs.T).squeeze()
        p_tmp = torch.nn.functional.softmax(cossim_tmp[idx_related], dim=0)
        kl_div = (p_prev*(p_prev.log() - p_tmp.log())).sum().detach().cpu()
        kl_divs[i] += kl_div

    idx_final = kl_divs.argsort()[0].item()

    return questions[idx_final]


def generate_answer(img_path="", query="", model_caps=None, processor_caps=None):
    img = Image.open(img_path)
    prompt = "Question: " + query + " Answer:"
    inputs_ = processor_caps(images=img, text=prompt, return_tensors='pt').to(device)
    out = model_caps.generate(**inputs_, do_sample=False)
    answer_generated = processor_caps.decode(out[0], skip_special_tokens=True).strip()

    return answer_generated


def paraphrase(text="", model='gpt-3.5-turbo-0613'):
    messages = [{"role": "system",
                 "content": "Your role is to paraphrase the given text into a fluent and natural text while preserving the information in the given text. Do not add your internal knowledge while paraphrasing."}]
    # messages = [{"role": "system", "content": "Your role is to paraphrase the given text."}]
    messages.append({"role": "user", "content": f"Text: {text}\nParaphrased: "})
    retry_count = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model,
                                                    messages=messages,
                                                    n=1,
                                                    temperature=0.7,
                                                    top_p=0.8,
                                                    request_timeout=10,
                                                    max_tokens=512)
            break
        except Exception as e:
            print(e)
            time.sleep(3 * (retry_count + 1))
            retry_count += 1
            if retry_count > 5:
                print("Too many retrials for generation, return the input")
                return text
            continue
    paraphrased = response['choices'][0]['message']['content']

    return paraphrased


dial_new = []
dial_new_recon = []

for idx in tqdm(range(s_idx, e_idx)):
    dial_gen = [visdial[idx]['dialog'][0]]
    img_path = './ChatIR/' + visdial[idx]['img']
    img_url = 'http://images.cocodataset.org/' + visdial[idx]['img']
    dial_new.append({})
    dial_new_recon.append({})
    dial_new[-1]['img'] = visdial[idx]['img']
    dial_new_recon[-1]['img'] = visdial[idx]['img']
    caption_recon = dial_gen[0]
    captions_recon = [caption_recon]

    for rnd in range(10):
        questions = []

        if referring:
            ques_prior = []
            prompt_refer, prompt_fake, top_k, cos_sims = get_referring_prompt(caption_recon, img_embs,
                                                                              recall_hitting, rnd + 1, search_space)
            for k in range(q_n):
                if filtering:
                    for _ in range(3):
                        response = generate_questions_referring(descrip=dial_gen, prompt_fake=prompt_fake,
                                                                n=1, ques_prior=questions + ques_prior)
                        output = response['choices'][0]['message']['content']
                        ques = output.split('?')[0] + '?'
                        ans = filter_questions(caption_recon, ques)

                        if 'uncertain' not in ans:
                            ques_prior.append(ques)
                        else:
                            break
                    if len(ques_prior) == 3:
                        response = generate_questions(dial_gen, n=1)
                        output = response['choices'][0]['message']['content']
                        ques = output.split('?')[0] + '?'
                        ans = filter_questions(caption_recon, ques)
                        if 'uncertain' not in ans:
                            ques = 'what is the other object in the image?'

                else:
                    response = generate_questions_referring(descrip=dial_gen, prompt_fake=prompt_fake,
                                                            n=1, ques_prior=questions + ques_prior)
                    ques = response['choices'][0]['message']['content'].split('?')[0] + '?'

                questions.append(ques)


        else:
            img_paths, top_k, cos_sims = search_imgs(caption_recon, img_embs, search_space, k=recall_hitting)
            prompt_fake = ""
            prompt_refer = ""
            for i in range(q_n):
                questions_tmp = generate_questions(dial_gen, n=1)
                questions.append(questions_tmp['choices'][i]['message']['content'].split('?')[0] + '?')

        if select:
            if rnd < 99:
                question_final = select_question(caption_recon=caption_recon,
                                                 questions=questions,
                                                 cossim_prev=cos_sims,
                                                 k=recall_hitting,
                                                 img_embs=img_embs,
                                                 threshold=threshold_low,
                                                 round=rnd+1)
            else:
                question_final = questions[0]
        else:
            question_final = questions[0]

        answer_generated = generate_answer(img_path=img_path,
                                           query=caption_recon + '. ' + question_final,
                                           model_caps=blip2,
                                           processor_caps=processor2)
        qa = question_final + " " + answer_generated
        dial_gen.append(qa)

        if reconstruct:
            caption_recon = reconstruct_dialog(dial_gen)
            if caption_recon == captions_recon[-1]:
                caption_recon = paraphrase(caption_recon)
        else:
            caption_recon = ', '.join(dial_gen)

        captions_recon.append(caption_recon)

    dial_new[-1]['dialog'] = dial_gen
    dial_new_recon[-1]['dialog'] = captions_recon

save_path = './ChatIR/dialogues/ours_final_'
save_path += 'q_n_' + str(q_n) + '_'
save_path += 'recall_hitting_' + str(recall_hitting) + '_'
save_path += 'thres_low_' + str(threshold_low) + '_'
save_path += 'recon_' +str(reconstruct).lower() + '_'
save_path += 'referring_' +str(referring).lower() + '_'
save_path += 'filtering_' +str(filtering).lower() + '_'
save_path += 'select_' +str(select).lower() + '_'
save_path_dial = save_path + str(s_idx) + '_' + str(e_idx) + '.json'
save_path_recon = save_path + 'reconed_' + str(s_idx) + '_' + str(e_idx) + '.json'

with open(save_path_dial, 'w') as f:
    json.dump(dial_new, f)

with open(save_path_recon, 'w') as f:
    json.dump(dial_new_recon, f)

print(f'saved dialog to the {save_path_dial}')
print(f'saved dialog reconstruction to the {save_path_recon}')
