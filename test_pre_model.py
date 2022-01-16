from transformers import AutoTokenizer, AutoModel ,RoFormerModel, RoFormerTokenizer, XLNetModel, XLNetTokenizer, AutoModelForMaskedLM, AutoTokenizer, AutoModel
import torch

if __name__ == "__main__":
    text = "j2：如果你也认为g2有嫌疑，那我就有把握了，他只是个数学老师，作案就一定会留下破绽。"
    """
    bert-base-chinese
    chinese-roberta-wwm-ext
    roformer_chinese_base
    chinese-xlnet-base
    """
    model_name = "chinese-xlnet-base"
    path = "./pre_models/" + model_name
    # if model_name == "roformer_chinese_base":
    #     tokenizer = RoFormerTokenizer.from_pretrained(path)
    #     model = RoFormerModel.from_pretrained(path)
    # elif model_name == "chinese-xlnet-base":
    #     tokenizer = AutoTokenizer.from_pretrained(path)
    #     model = AutoModel.from_pretrained(path)
    # elif model_name == "chinese-roberta-wwm-ext":
    #     tokenizer = AutoTokenizer.from_pretrained(path)
    #     model = AutoModel.from_pretrained(path)

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, output_hidden_states=True, return_dict=True)

    inputs = tokenizer.encode_plus(text,
                                    truncation=True,
                                    max_length=100,
                                    pad_to_max_length=True,
                                    return_tensors='pt'
                                    )
    input = inputs['input_ids'].squeeze(0)
    attention_mask = inputs['attention_mask'].squeeze(0)
    token_type_ids = inputs["token_type_ids"].squeeze(0)
    outputs = model(input_ids=input, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # inputs = tokenizer(text, return_tensors="pt")
    # outputs = model(**inputs)


    print("--")
