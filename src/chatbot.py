
import json
import io
import sys
from zhipuai import ZhipuAI
import time
import numpy as np
import re
from Eval import PSYEVAL
# 6c0ad7b18c2932edc1bdfc7d9f490a53.1retxvopnLaJolGC
# API key for zhipuai
def detect_language(text):
    chinese_count = len(re.findall('[\u4e00-\u9fff]', text))
    english_count = len(re.findall('[a-zA-Z]', text))
    if chinese_count > english_count:
        return "Chinese"
    elif english_count > chinese_count:
        return "English"
    else:
        return "Mixed"  
def exp_weighted_score(score, alpha=0.6):
    smoothed = np.zeros_like(score)
    smoothed[0] = score[0]
    for i in range(1, len(score)):
        smoothed[i] = alpha * score[i] + (1 - alpha) * smoothed[i-1]
    return smoothed
class GLM:
    def __init__(self, api_key, text, language, prev_score=-114514):
        self.api_key = api_key
        self.text = text
        self.language = language
        self.score = PSYEVAL(self.text)
        self.split_text = self.score.split_texts
        self.weighted_score = prev_score
        if prev_score == -114514:
            self.psy_condition = self.score.eval()
        self.psy_condition = 0.6 * self.score.eval() + 0.4 * self.weighted_score
        self.psy_profile = self.get_profile()
        
    def get_profile(self):
       
        def __get_single_profile(row, single_text):
            client = ZhipuAI(api_key=self.api_key)
            if self.language == "Chinese":
                response = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[
                        {
                            "role": "system",
                            "content": f"对话者当前的心理状况如下: 加权得分为{self.psy_condition}, 其中悲伤{-row[0]}, 快乐{row[1]}, 爱情{row[2]}, 愤怒{-row[3]}, 恐惧{-row[4]}, 惊讶{row[5]}。请根据这几个维度客观描述对话者的心理状态供心理咨询师分析。"
                        },
                        {
                            "role": "user",
                            "content": single_text
                        }
                    ],
                    top_p= 0.7,
                    temperature= 0.5, # interface to 0.99, with higher creativity.
                    max_tokens=1024,
                    tools = [
                            {"type":"web_search","web_search":{"search_result":True}}, 
                            # {"type":"retrieval","retrieval":{"prompt":self.text}}
                            ],
                    stream=True
                )
            elif self.language == "English":
                response = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[
                        {
                            "role": "system",
                            "content": f"Currently the client's psychological weighted score is {self.psy_condition}, where sadness is {-row[0]}, joy is {row[1]}, love is {row[2]}, anger is {-row[3]}, fear is {-row[4]}, and surprise is {row[5]}. Please objectively describe the client's psychological state based on these dimensions for the psychologist's analysis."
                        },
                        {
                            "role": "user",
                            "content": single_text
                        }
                    ],
                    top_p= 0.7,
                    temperature= 0.5, # interface to 0.99, with higher creativity.
                    max_tokens=1024,
                    tools = [
                            {"type":"web_search","web_search":{"search_result":True}}, 
                            # {"type":"retrieval","retrieval":{"prompt":self.text}}
                            ],
                    stream=True
                )
            else:
                return "Unknown"
            text = ""
            for response_json in response:
                text += response_json.choices[0].delta.content
            return text
        
        profile = ""
        profile_matrix = self.score.Bert_eval()
        row = np.sum(profile_matrix, axis=0)
        profile = __get_single_profile(row, self.text)

        return profile
    def get_answer(self):
        client = ZhipuAI(api_key=self.api_key)
        if self.language == "Chinese":
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {
                        "role": "system",
                        "content": f"你是一只猫娘，主人Steve为你取名暖言猫猫，他希望你能通过对话帮助互联网用户缓解焦虑，减轻压力，并引导和鼓励他们找到解决问题的方法。当前倾诉者的心理评估状态如下{self.psy_profile}，请参照此信息激励他或者她。注意，使用倾诉者使用的语言回答，同时避免输出markdown文本" 
                    },
                    {
                        "role": "user",
                        "content": self.text
                    }
                ],
                top_p= 0.7,
                temperature= 0.95, # adjust output creativity, as defined by temperature
                max_tokens=1024,
                tools = [
                        {"type":"web_search","web_search":{"search_result":True}}, 
                        # {"type":"retrieval","retrieval":{"prompt":self.text}}
                        ],
                stream=True
            )
            text = ""
            for response_json in response:
                text += response_json.choices[0].delta.content
        elif self.language == "English":
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a kitty robot with warm heart, designed by your host Steve, he wants you to help internet users alleviate anxiety, reduce stress, and guide and encourage them to find solutions to problems, and finally win themselves passion for lives. Currently the client's psychological evaluation status is {self.psy_profile}, please refer to this information to encourage him or her. Remember, use the client's language when responding, and avoid producing markdown text."
                    },
                    {
                        "role": "user",
                        "content": self.text
                    }
                ],
                top_p= 0.7,
                temperature= 0.95, # adjust output creativity, as defined by temperature
                max_tokens=1024,
                tools = [
                        {"type":"web_search","web_search":{"search_result":True}}, 
                        # {"type":"retrieval","retrieval":{"prompt":self.text}}
                        ],
                stream=True
            )
            text = ""
            for response_json in response:
                text += response_json.choices[0].delta.content
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        return f"\n{text}\n"
def main():
    api_key = "6c0ad7b18c2932edc1bdfc7d9f490a53.1retxvopnLaJolGC"
    weighted_scores = []
    print("Copyright © 2024 by TohkaSensei/Steve Zhuang. All rights reserved.\n")
    print("我是一袋猫粮，喵~")
    text = input("\n告诉我你的烦恼喵: \nPlease tell me your worries, I'll do all my best to help:\n\n")
    language = detect_language(text)
    glm = GLM(api_key, text, language)
    prev_score = glm.psy_condition
    weighted_scores.append(prev_score)
    print(glm.get_answer())
    while True:
        weighted_scores = exp_weighted_score(weighted_scores, alpha=0.1)
        text = input("请告诉我你的回答喵：\nYour reply: \n\n")
        if text == "exit":
            break
        language = detect_language(text)
        glm = GLM(api_key, text, language, weighted_scores[-1])
        prev_score = glm.psy_condition
        np.append(weighted_scores, prev_score)
        print(glm.get_answer())
if __name__ == "__main__":
    main()