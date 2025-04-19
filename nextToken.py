# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "textual",
#     "torch",
#     "transformers",
# ]
# ///



import torch
from textual.app import App, ComposeResult
from textual.widgets import Input, Button, Static, Switch, DataTable, Label, Header, Footer
from textual.reactive import reactive
from textual.containers import ScrollableContainer, Horizontal
from textual import on
from textual.binding import Binding
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
# Auto select device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device);

class HelloNameApp(App):
    CSS = """
ScrollableContainer {
    margin: 1 5 1 5;
    max-width: 80%;
}


.container {
    margin: 0 5 0 5;
    height: auto;
    width: auto;
}

.label {
    height: 3;
    content-align: center middle;
    width: auto;
}

Switch {
    border: solid green;
    height: auto;
    width: auto;
}

Input {
    max-width: 80%;
}

Label {
    margin: 1 5 1 5;
    color: $primary;
    max-width: 80%;
}

Button {
    margin: 0 2 0 0;
}

DataTable {
    max-width: 80%;
}
"""
    BINDINGS = [
        Binding("d", "toggle_dark", "Toggle dark mode", show=True),
        ("right,l", "append_token", "Add"),
        ("left,h", "pop_token", "Remove"),
        ("down,j", "cursor_down", "Cursor down"),
        ("up,k", "cursor_up", "Cursor up"),
        ("ctrl+q", "quit", "Quit")
    ]
    system_prompt = reactive("")
    user_prompt = reactive("")
    length_prompt = reactive(0)
    prompt = reactive("")
    add_full_response = reactive(True)
    
    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            yield Header()
            yield Footer()
            yield Input(placeholder="Write a system prompt (optional)", id="system_prompt")
            yield Input(placeholder="Write a user prompt (optional)", id="user_prompt").focus()
            with Horizontal(classes="container"):
                yield Button("Submit", variant="primary")
                yield Static("Add full reponse:", classes="label")
                yield Switch(value=True)
            self.label = Label(f"[yellow]{self.prompt[self.length_prompt:]}[/yellow]")
            yield self.label
            yield DataTable()
        
    def on_mount(self) -> None:

        self.title = "Next token prediction"

        def update_prompt(self) -> None:
            messages = []
            if self.system_prompt != "":
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": self.user_prompt})
            self.prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self.length_prompt = len(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            self.query_one(Label).update(f"[yellow]{self.prompt[self.length_prompt:]}[/yellow]")
        
        def update_system_prompt(value: str) -> None:
            self.system_prompt = value
            update_prompt(self)
            
        def update_user_prompt(value: str) -> None:
            self.user_prompt = value
            update_prompt(self)

        self.watch(self.query_one("#system_prompt", Input), "value", update_system_prompt)
        self.watch(self.query_one("#user_prompt", Input), "value", update_user_prompt)

    def update_table(self, prompt: str) -> None:
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor([tokens]).to(device)
        with torch.no_grad():
            outputs = model(tokens)
        logits = outputs.logits[0]
        probs = torch.softmax(logits[-1], dim=0)
        top_k = 22
        top_probs, top_indices = torch.topk(probs, top_k)
        ROWS = []
        for prob, idx in zip(top_probs, top_indices):
            token = tokenizer.decode([idx])
            ROWS.append((token, idx.item(), f"{prob:.3f}"))
        table = self.query_one(DataTable)
        table.zebra_stripes = True
        table.clear(columns=True)
        table.add_columns("token","token_id","probability")
        table.add_rows(ROWS)
        table.cursor_type = "row"
        table.focus()
        
    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

    def action_append_token(self) -> None:
        table = self.query_one(DataTable)
        table.zebra_stripes = True
        # Get the keys for the row under the cursor.
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        if row_key is not None:
            token_id = table.get_row(row_key)[1]
            token = tokenizer.decode([token_id])
            self.prompt += token
            self.query_one(Label).update(f"{self.prompt[self.length_prompt:]}")
            self.update_table(f"{self.prompt}")

    def action_pop_token(self) -> None:
        tokens_without_last = tokenizer.encode(self.prompt)[:-1]
        print(len(tokens_without_last))
        print(self.length_prompt)
        if len(tokens_without_last)>0:
            self.prompt = tokenizer.decode(tokens_without_last)
            self.query_one(Label).update(f"{self.prompt[self.length_prompt:]}")
            self.update_table(f"{self.prompt}")

    def action_cursor_down(self) -> None:
        table = self.query_one(DataTable)
        table.zebra_stripes = True
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        current_row_index = table.get_row_index(row_key)
        table.move_cursor(row=current_row_index+1)

    def action_cursor_up(self) -> None:
        table = self.query_one(DataTable)
        table.zebra_stripes = True
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        current_row_index = table.get_row_index(row_key)
        table.move_cursor(row=current_row_index-1)

    @on(Switch.Changed)
    def switch(self, event: Switch.Changed):
        self.add_full_response = not self.add_full_response

    @on(Button.Pressed)
    def update_the_prompt(self):
        messages = []
        if self.system_prompt != "":
            messages.append({"role": "system", "content": f"{self.system_prompt}"})
        messages.append({"role": "user", "content": f"{self.user_prompt}"})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.add_full_response:
            model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
            prompt_length = model_inputs['input_ids'].shape[1]
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                eos_token_id= tokenizer.eos_token_id,
            )
            self.prompt = prompt + tokenizer.decode(outputs[0][prompt_length:])
        else:
            self.prompt = prompt
        self.query_one(Label).update(f"[bright_white]{self.prompt[self.length_prompt:]}[/bright_white]")
        self.update_table(f"{self.prompt}")

if __name__ == "__main__":
    app = HelloNameApp()
    app.run()
