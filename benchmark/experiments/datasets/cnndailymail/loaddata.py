from datasets import load_dataset

# prompt = "Please summarize the following text, and highlight the most important part: "

prompt = "Please generate a detailed and comprehensive response to the following request. Cover all possible aspects and provide multiple examples. Break down the topic into subtopics if needed, and avoid omitting any relevant details."

def loaddata(num_sequence, min_len=0, max_len=2000, return_sum_length=False):
    ds = load_dataset("abisee/cnn_dailymail", "1.0.0", trust_remote_code=True, split="test")
    res = []
    lens_list = []
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    i = 0
    while len(res) < num_sequence:
        txt = ds[i]["article"]
        i += 1
        content = prompt + txt
        chat_input = [{"role": "user", "content": content}]
        content_after_template = tokenizer.apply_chat_template(chat_input, tokenize=False)
        inputs_id_len = len(tokenizer.encode(content_after_template))
        if inputs_id_len > max_len or inputs_id_len < min_len:
            continue
        res.append(content)
        lens_list.append(inputs_id_len)
    if return_sum_length:
        return res, sum(lens_list)
    else:
        return res

if __name__ == "__main__":
    sequences, sum_length = loaddata(400, min_len=1000, max_len=2000, return_sum_length=True)
    # print(sequences)
    print(sequences[340])
    print(len(sequences))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    sum_length = 0
    max_length = 0
    for seq in sequences:
        input_ids = tokenizer.encode(seq)
        sum_length += len(input_ids)
        max_length = max(max_length, len(input_ids))
    print(f"sequence的平均长度为: {sum_length / len(sequences):.2f}")
    print(f"sequence的最大长度为: {max_length}")
