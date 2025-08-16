from datasets import load_dataset

# prompt = "Please solve the following problem in online judge, write the detailed solution and C++ code. The output structure should be: {Problem Analysis, Problem Solution, Code, Code Analysis(must be very detailed)}. The problem is: \n"
# prompt = "Please solve the following problem in online judge, write the detailed solution and C++ code. The output structure should be: {Problem Simple Description, Problem Analysis, Problem Solution, Code, Code Analysis(must be very detailed)}. The problem is: \n"
# prompt = "Please solve the following problem in online judge, write the detailed solution and C++ code. The output structure should be: {Problem Solution, Code, Code Analysis(must be very detailed)}. The problem is: \n"
prompt = "Please provide a comprehensive solution to the following programming problem from an online judge platform, write the detailed solution and C++ code. If the code has more than one solution, write all soluaiton. The output structure should be: {Solution Id[Solution id], Code[complete C++ implementation], Code Analysis[detailed technical breakdown of the solution]}. The problem is: \n"
# prompt = "Please provide a comprehensive solution to the following programming problem from an online judge platform. Your response should include:\n 1.Problem Analysis: A detailed explanation of the problem requirements, input/output specifications, and any edge cases to consider.\n 2.Solution Approach: Describe all possible solution strategies with their time and space complexity analysis.\n 3. Code Implementation: Provide well-structured, commented C++ code for each viable solution; Include proper input handling and edge case management; Follow best coding practices (proper naming, modularity, etc.).\n 4. Code Analysis: For each solution, provide a detailed breakdown of: Key algorithmic components; Time and space complexity with justification; Critical code segments and their functionality.\n"

def loaddata(num_sequence, min_len=0, max_len=2000, return_sum_length=False):
    res = []
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    ds_list = [load_dataset("codeparrot/apps", split="test", difficulties=["competition"]), load_dataset("codeparrot/apps", split="train", difficulties=["competition"]), load_dataset("codeparrot/apps", split="test", difficulties=["interview"]), load_dataset("codeparrot/apps", split="train", difficulties=["interview"]), load_dataset("codeparrot/apps", split="test", difficulties=["introductory"]), load_dataset("codeparrot/apps", split="train", difficulties=["introductory"])]
    lens_list = []
    for ds in ds_list:
        if len(res) >= num_sequence:
            break
        for i in range(len(ds)):
            
            txt = ds[i]["question"]
            i += 1
            content = prompt + txt
            chat_input = [{"role": "user", "content": content}]
            content_after_template = tokenizer.apply_chat_template(chat_input, tokenize=False)
            inputs_id_len = len(tokenizer.encode(content_after_template))
            if inputs_id_len > max_len or inputs_id_len < min_len:
                continue
            res.append(content)
            lens_list.append(inputs_id_len)
            if len(res) >= num_sequence:
                break
    if len(res) < num_sequence:
        res += res[:num_sequence-len(res)]
        lens_list += lens_list[:num_sequence-len(lens_list)]
    if return_sum_length:
        return res, sum(lens_list)
    else:
        return res

if __name__ == "__main__":
    # sequences = loaddata(3000, min_len=128, max_len=500)
    sequences = loaddata(1500, min_len=1000, max_len=2000)
    # print(sequences[0])
    # print("-"*100)
    # print(sequences[1])
    # print("-"*100)
    # print(sequences[2])
    # print("-"*100)
    print(len(sequences))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    sum_length = 0
    max_length = 0
    for seq in sequences:
        chat_input = [{"role": "user", "content": seq}]
        prompt = tokenizer.apply_chat_template(chat_input, tokenize=False)
        input_ids = tokenizer.encode(prompt)
        sum_length += len(input_ids)
        max_length = max(max_length, len(input_ids))
    print(f"sequence的平均长度为: {sum_length / len(sequences):.2f}")
    print(f"sequence的最大长度为: {max_length}")
'''
500-1000: avg: 685
1000-2000, avg: 1189.11
2000-3500, avg: 2410.50
'''