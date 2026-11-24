import torch
from torch import nn

from .visual_tokenizer import VisualTokenizer

class GeneratorEncoderInputs(nn.Module):
    """Build the input embeddings and the attention mask for the T5 encoder."""

    def __init__(self, image_channels: int, embedding_dim: int):
        super().__init__()
        self.visual_tokenizer = VisualTokenizer(image_channels, embedding_dim)

    def forward(self, image_features: torch.Tensor, prompt_embeddings: torch.Tensor, prompt_attention_mask: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        # Both sequences have shape [batch, sequence length, embedding width]
        # Join aling the sequence dimensions so that T5 can attend to image and text together
        visual_tokens = self.visual_tokenizer(image_features)
        inputs_embeds = torch.cat((visual_tokens, prompt_embeddings), dim=1)

        # Every visual token is valid, keep the prompt's original padding mask
        visual_attention_mask = prompt_attention_mask.new_ones(visual_tokens.shape[:2])
        attention_mask = torch.cat((visual_attention_mask, prompt_attention_mask), dim=1)

        return inputs_embeds, attention_mask

class CXRReportGenerator(nn.Module):
    """Connect frozen DenseNet image features to a T5 report decoder."""

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor

    def __init__(self, densenet, t5, tokenizer, prompt: str):
        super().__init__()
        self.densenet = densenet
        self.t5 = t5
        self.encoder_inputs = GeneratorEncoderInputs(image_channels=1024, embedding_dim=t5.config.d_model,)

        # Reuse the pretrained image encoder without updating its weights or its batch-normalization statistics.
        self.densenet.requires_grad_(False)
        self.densenet.eval()

        # Tokenize the fixed prompt once. Buffers move with the model when we change devices. We can recreate them from the saved prompt.
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        self.register_buffer("prompt_ids", prompt_tokens["input_ids"], persistent=False)
        self.register_buffer("prompt_mask", prompt_tokens["attention_mask"], persistent=False)

    def train(self, mode: bool = True):
        # PyTorch switches all child modules into training mode by default.
        # Keep frozen DenseNet in eval mode even while the generator trains.
        
        super().train(mode)
        self.densenet.eval()
        return self

    def prepare_encoder_inputs(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Take the spatial feature map before DenseNet averages the grid.
        # Gradients resume after this block so the visual projection can learn

        with torch.no_grad():
            image_features = self.densenet.features(images).relu()

        batch_size = images.shape[0]
        prompt_ids = self.prompt_ids.expand(batch_size, -1)
        prompt_mask = self.prompt_mask.expand(batch_size, -1)
        prompt_embeddings = self.t5.get_input_embeddings()(prompt_ids)

        return self.encoder_inputs(image_features, prompt_embeddings, prompt_mask)

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        inputs_embeds, attention_mask = self.prepare_encoder_inputs(images)

        # T5 shifts the target IDs internally for teacher forcing, then compares its predictions with the original target IDs.
        return self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
    
if __name__ == "__main__":
    encoder_inputs = GeneratorEncoderInputs(image_channels=1024, embedding_dim=512)

    image_features = torch.randn(2, 1024, 7, 7)
    prompt_embeddings = torch.randn(2, 5, 512)

    # Simulate a four-token prompt padded to five slots for each image.
    prompt_attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])

    with torch.no_grad():
        inputs_embeds, attention_mask = encoder_inputs(image_features, prompt_embeddings, prompt_attention_mask)

    # Each image gets 16 visual tokens followed by 5 prompt slots
    assert inputs_embeds.shape == (2, 21, 512)
    assert attention_mask.shape == (2, 21)
    assert torch.equal(inputs_embeds[:, 16:], prompt_embeddings)
    assert torch.all(attention_mask[:, :16] == 1)
    assert torch.equal(attention_mask[:, 16:], prompt_attention_mask)

    print(f"Encoder embeddings: {tuple(inputs_embeds.shape)}")
    print(f"Attention mask: {tuple(attention_mask.shape)}")
    print(f"Visible tokens per example: {attention_mask.sum(dim=1).tolist()}")