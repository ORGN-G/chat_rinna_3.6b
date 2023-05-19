import tkinter as tk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft")

if torch.cuda.is_available():
    model = model.to("cuda")

# 最初のpromptを指定
prompt = [
    {
        "role": "りんな",
        "content": "私はりんなだよ♪高校生です！なんでも質問してね！"
    },
    {
        "role": "user",
        "content": "りんなはチャットAIですか？"
    },
    {
        "role": "りんな",
        "content": "私はAIじゃないよ！君といっぱいお話したいな"
    }
]
utterances = []


def generate_response():
    global prompt
    global utterances
    user_input = user_input_entry.get()

    # 入力した文字列を表示
    response_text.config(state=tk.NORMAL)
    response_text.insert(tk.END, user_input + "\n")
    response_text.config(state=tk.DISABLED)

    # Promptにユーザーの入力を追加
    prompt.append({"role": "user", "content": user_input})

    character_count = sum(len(item["content"]) for item in prompt)
    if character_count > 300:
        prompt = prompt[1:]
        utterances = []

    utterances = [f"{uttr['role']}: {uttr['content']}" for uttr in prompt]
    input_text = "<NL>".join(utterances) + "<NL>りんな: "

    token_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=128,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    output = output.replace("<NL>", "\n")
    output_display = output.replace("</s>", "")
    response_text.config(state=tk.NORMAL)
    response_text.insert(tk.END, "りんな: " + output_display + "\n")
    response_text.see(tk.END)
    response_text.config(state=tk.DISABLED)

    user_input_entry.delete(0, tk.END)
    user_input_entry.focus_set()

    # Promptに生成された応答を追加
    output = output.replace("\n", "<NL>")
    prompt.append({"role": "りんな", "content": output})

#Enterキーで送信できるようにする
def handle_key_press(event):
    if event.keysym == "Return":  # Enter
        send_button.invoke()  # 送信ボタンを押す


root = tk.Tk()
root.title("japanese-gpt-neox-3.6b-instruction-sft")
root.geometry("500x400")  # ウィンドウサイズを指定

response_label = tk.Label(root, text="Chat:")
response_label.pack()

response_text = tk.Text(root, width=70, height=25)
response_text.config(state=tk.DISABLED)
response_text.pack()

user_input_frame = tk.Frame(root)
user_input_frame.pack()

user_input_label = tk.Label(user_input_frame, text="User Input:")
user_input_label.pack(side=tk.LEFT)

user_input_entry = tk.Entry(user_input_frame, width=40)
user_input_entry.pack(side=tk.LEFT)
root.bind("<KeyPress-Return>", handle_key_press)  # Enterのハンドリング

send_button = tk.Button(user_input_frame, text="Send", command=generate_response)
send_button.pack(side=tk.LEFT)

root.mainloop()
