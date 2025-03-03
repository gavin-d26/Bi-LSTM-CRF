#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharCNN(nn.Module):
    """
    Character-level CNN for word representations
    """

    def __init__(
        self, char_vocab_size, char_emb_dim, char_hidden_dim, filter_sizes, dropout
    ):
        super(CharCNN, self).__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(char_emb_dim, char_hidden_dim, kernel_size=k)
                for k in filter_sizes
            ]
        )
        self.filter_sizes = filter_sizes
        self.char_hidden_dim = char_hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_indices):
        """
        Args:
            char_indices: (batch_size, seq_len, max_word_len)
        Returns:
            char_embeds: (batch_size, seq_len, char_hidden_dim)
        """
        batch_size, seq_len, max_word_len = char_indices.size()

        # Reshape to process all words in the batch together
        char_indices = char_indices.view(
            -1, max_word_len
        )  # (batch_size * seq_len, max_word_len)

        # Embed characters
        char_embeds = self.char_embedding(
            char_indices
        )  # (batch_size * seq_len, max_word_len, char_emb_dim)

        # Transpose for Conv1d which expects (batch, channels, length)
        char_embeds = char_embeds.transpose(
            1, 2
        )  # (batch_size * seq_len, char_emb_dim, max_word_len)

        # Apply convolutions and max pooling
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            # Check if input length is smaller than kernel size
            kernel_size = self.filter_sizes[i]
            if max_word_len < kernel_size:
                # Fallback for very short sequences: use adaptive pooling
                # Create a zero tensor with the expected output shape
                dummy_output = torch.zeros(
                    char_embeds.size(0),
                    self.char_hidden_dim,
                    device=char_indices.device,
                )
                conv_outputs.append(dummy_output)
                continue

            # Proceed with normal convolution for inputs longer than kernel_size
            conv_output = F.relu(
                conv(char_embeds)
            )  # (batch_size * seq_len, char_hidden_dim, max_word_len - k + 1)

            # Handle global max pooling
            conv_output = F.max_pool1d(
                conv_output, conv_output.size(2)
            )  # (batch_size * seq_len, char_hidden_dim, 1)

            conv_outputs.append(
                conv_output.squeeze(2)
            )  # (batch_size * seq_len, char_hidden_dim)

        # Concatenate outputs from different filters
        char_embeds = torch.cat(
            conv_outputs, dim=1
        )  # (batch_size * seq_len, char_hidden_dim * num_filters)

        # Reshape back to batch format
        char_embeds = char_embeds.view(
            batch_size, seq_len, -1
        )  # (batch_size, seq_len, char_hidden_dim * num_filters)

        return self.dropout(char_embeds)


class BiLSTMCRF(nn.Module):
    """
    Basic BiLSTM-CRF model for sequence labeling
    """

    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.5,
        use_char_cnn=False,
        char_params=None,
    ):
        super(BiLSTMCRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.use_char_cnn = use_char_cnn

        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Character CNN
        self.char_hidden_dim = 0
        if use_char_cnn:
            assert char_params is not None, "Character CNN parameters must be provided"
            char_vocab_size = char_params["char_vocab_size"]
            char_emb_dim = char_params["char_emb_dim"]
            self.char_hidden_dim = char_params["char_hidden_dim"] * len(
                char_params["filter_sizes"]
            )
            self.char_cnn = CharCNN(
                char_vocab_size=char_vocab_size,
                char_emb_dim=char_emb_dim,
                char_hidden_dim=char_params["char_hidden_dim"],
                filter_sizes=char_params["filter_sizes"],
                dropout=dropout,
            )

        # BiLSTM
        lstm_input_dim = embedding_dim + self.char_hidden_dim
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim // 2,  # Bidirectional will double this
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Maps the LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # CRF layer parameters
        # Transition matrix: transitions[i, j] = score of transitioning from j to i
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))

        # These transitions are impossible and set to a large negative value
        # Cannot transition from any tag to START tag
        # Cannot transition from END tag to any tag
        self.transitions.data[:, 0] = -10000  # No transitions to START_TAG
        self.transitions.data[0, :] = -10000  # No transitions from END_TAG

    def forward(self, word_indices, char_indices=None, seq_lengths=None, mask=None):
        """
        Get LSTM features
        Args:
            word_indices: (batch_size, max_seq_len)
            char_indices: (batch_size, max_seq_len, max_word_len)
            seq_lengths: (batch_size)
            mask: (batch_size, max_seq_len)
        Returns:
            lstm_feats: (batch_size, max_seq_len, tagset_size)
        """
        batch_size, max_seq_len = word_indices.size()

        # Create mask if not provided
        if mask is None:
            mask = torch.ones_like(word_indices, dtype=torch.uint8)
            for i, length in enumerate(seq_lengths):
                mask[i, length:] = 0

        # Word embeddings
        word_embeds = self.word_embeddings(
            word_indices
        )  # (batch_size, max_seq_len, embedding_dim)
        word_embeds = self.dropout(word_embeds)

        # Concatenate with character embeddings if using char CNN
        if self.use_char_cnn and char_indices is not None:
            char_embeds = self.char_cnn(
                char_indices
            )  # (batch_size, max_seq_len, char_hidden_dim)
            embeds = torch.cat(
                [word_embeds, char_embeds], dim=2
            )  # (batch_size, max_seq_len, embedding_dim + char_hidden_dim)
        else:
            embeds = word_embeds

        # Pack padded sequence for LSTM
        packed_embeds = pack_padded_sequence(
            embeds, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # LSTM forward pass
        packed_lstm_out, _ = self.lstm(packed_embeds)

        # Unpack sequence
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)

        # Map to tag space
        lstm_feats = self.hidden2tag(lstm_out)  # (batch_size, max_seq_len, tagset_size)

        return lstm_feats

    def _score_sentence(self, lstm_feats, tags, mask):
        """
        Calculate the score of a given sequence of tags
        Args:
            lstm_feats: (batch_size, max_seq_len, tagset_size)
            tags: (batch_size, max_seq_len)
            mask: (batch_size, max_seq_len)
        Returns:
            score: (batch_size)
        """
        batch_size, max_seq_len, _ = lstm_feats.size()

        # Get the emission scores for the given tags
        score = torch.zeros(batch_size, device=lstm_feats.device)

        # Add emission scores
        for t in range(max_seq_len):
            mask_t = mask[:, t]
            emit_scores_t = lstm_feats[range(batch_size), t, tags[:, t]]
            score += emit_scores_t * mask_t

        # Add transition scores
        for t in range(max_seq_len - 1):
            mask_t = mask[:, t + 1]
            trans_scores_t = self.transitions[tags[:, t + 1], tags[:, t]]
            score += trans_scores_t * mask_t

        return score

    def _forward_alg(self, lstm_feats, mask):
        """
        Calculate the partition function with the forward algorithm
        Args:
            lstm_feats: (batch_size, max_seq_len, tagset_size)
            mask: (batch_size, max_seq_len)
        Returns:
            alpha: (batch_size)
        """
        batch_size, max_seq_len, tagset_size = lstm_feats.size()

        # Initialize forward variables
        init_alphas = torch.full(
            (batch_size, tagset_size), -10000.0, device=lstm_feats.device
        )
        init_alphas[:, 0] = 0.0  # START_TAG has all the score

        # Forward variables
        forward_var = init_alphas

        # Iterate through the sentence
        for t in range(max_seq_len):
            # Broadcast forward_var to match transitions
            emit_scores = lstm_feats[:, t].unsqueeze(2)  # (batch_size, tagset_size, 1)

            # Calculate all possible tag scores at timestep t
            # forward_var: (batch_size, tagset_size)
            # self.transitions: (tagset_size, tagset_size)
            # next_tag_var: (batch_size, tagset_size, tagset_size)
            next_tag_var = forward_var.unsqueeze(1) + self.transitions

            # Sum over all previous tags
            viterbivars_t = next_tag_var + emit_scores

            # Apply log-sum-exp trick for numerical stability
            forward_var = torch.logsumexp(viterbivars_t, dim=2)

            # If mask is 0, keep the previous value
            mask_t = mask[:, t].unsqueeze(1)
            forward_var = mask_t * forward_var + (1 - mask_t) * forward_var

        # Terminal transition score
        # No terminal state for now

        # Final forward variable
        alpha = torch.logsumexp(forward_var, dim=1)  # (batch_size)

        return alpha

    def neg_log_likelihood(self, lstm_feats, tags, mask=None):
        """
        Calculate negative log likelihood loss
        Args:
            lstm_feats: (batch_size, max_seq_len, tagset_size)
            tags: (batch_size, max_seq_len)
            mask: (batch_size, max_seq_len)
        Returns:
            loss: scalar
        """
        forward_score = self._forward_alg(lstm_feats, mask)
        gold_score = self._score_sentence(lstm_feats, tags, mask)

        # Loss: partition function - score
        loss = torch.mean(forward_score - gold_score)

        return loss

    def max_margin_loss(self, lstm_feats, tags, mask=None, cost_factor=1.0):
        """
        Calculate max-margin (hinge) loss
        Args:
            lstm_feats: (batch_size, max_seq_len, tagset_size)
            tags: (batch_size, max_seq_len)
            mask: (batch_size, max_seq_len)
            cost_factor: scaling factor for the cost function
        Returns:
            loss: scalar
        """
        batch_size, max_seq_len, tagset_size = lstm_feats.size()
        gold_score = self._score_sentence(lstm_feats, tags, mask)

        # Find the sequence with the highest score (plus cost)
        viterbi_tags = self._viterbi_decode_batched(lstm_feats, mask)
        viterbi_tags = torch.LongTensor(viterbi_tags).to(lstm_feats.device)

        # Calculate Hamming loss (cost)
        hamming_cost = (
            torch.sum((viterbi_tags != tags).float() * mask, dim=1) * cost_factor
        )

        # Calculate score of predicted sequence
        pred_score = self._score_sentence(lstm_feats, viterbi_tags, mask)

        # Loss: max(0, margin + cost + pred_score - gold_score)
        margin = 1.0  # Fixed margin
        loss = torch.mean(
            torch.clamp(margin + hamming_cost + pred_score - gold_score, min=0)
        )

        return loss

    def softmax_margin_loss(self, lstm_feats, tags, mask=None, cost_factor=1.0):
        """
        Calculate softmax-margin loss
        Args:
            lstm_feats: (batch_size, max_seq_len, tagset_size)
            tags: (batch_size, max_seq_len)
            mask: (batch_size, max_seq_len)
            cost_factor: scaling factor for the cost function
        Returns:
            loss: scalar
        """
        batch_size, max_seq_len, tagset_size = lstm_feats.size()

        # TODO: Implement full softmax-margin loss with cost function
        # For now, we'll just use a simpler implementation

        # Get gold sequence score
        gold_score = self._score_sentence(lstm_feats, tags, mask)

        # Forward algorithm with cost-augmented potentials
        # For simplicity, we'll use a basic forward algorithm
        forward_score = self._forward_alg(lstm_feats, mask)

        # Loss: partition function - score
        loss = torch.mean(forward_score - gold_score)

        return loss

    def ramp_loss(self, lstm_feats, tags, mask=None, cost_factor=1.0):
        """
        Calculate ramp loss
        Args:
            lstm_feats: (batch_size, max_seq_len, tagset_size)
            tags: (batch_size, max_seq_len)
            mask: (batch_size, max_seq_len)
            cost_factor: scaling factor for the cost function
        Returns:
            loss: scalar
        """
        batch_size, max_seq_len, tagset_size = lstm_feats.size()

        # Get gold sequence score
        gold_score = self._score_sentence(lstm_feats, tags, mask)

        # Find the best scoring sequence with Viterbi
        viterbi_tags = self._viterbi_decode_batched(lstm_feats, mask)
        viterbi_tags = torch.LongTensor(viterbi_tags).to(lstm_feats.device)

        # Calculate Hamming loss (cost)
        hamming_cost = (
            torch.sum((viterbi_tags != tags).float() * mask, dim=1) * cost_factor
        )

        # Calculate score of predicted sequence
        pred_score = self._score_sentence(lstm_feats, viterbi_tags, mask)

        # Compute the cost-augmented scores
        cost_augmented_score = pred_score + hamming_cost

        # Compute loss: -max_y(score(x,y)) + max_y'(score(x,y') + cost(y,y'))
        loss = torch.mean(-gold_score + cost_augmented_score)

        return loss

    def soft_ramp_loss(self, lstm_feats, tags, mask=None, cost_factor=1.0):
        """
        Calculate soft ramp loss
        Args:
            lstm_feats: (batch_size, max_seq_len, tagset_size)
            tags: (batch_size, max_seq_len)
            mask: (batch_size, max_seq_len)
            cost_factor: scaling factor for the cost function
        Returns:
            loss: scalar
        """
        batch_size, max_seq_len, tagset_size = lstm_feats.size()

        # Get gold sequence score
        gold_score = self._score_sentence(lstm_feats, tags, mask)

        # Forward algorithm to compute log-sum-exp over all possible sequences
        forward_score = self._forward_alg(lstm_feats, mask)

        # Compute loss: -log(sum_y e^score(x,y)) + log(sum_y' e^(score(x,y') + cost(y,y')))
        # For simplicity, we approximate the second term with standard forward score
        # A complete implementation would require modifying forward algorithm to include costs

        loss = torch.mean(-torch.logsumexp(gold_score, dim=0) + forward_score)

        return loss

    def _viterbi_decode_batched(self, lstm_feats, mask):
        """
        Find the most likely tag sequence using Viterbi algorithm
        Args:
            lstm_feats: (batch_size, max_seq_len, tagset_size)
            mask: (batch_size, max_seq_len)
        Returns:
            best_paths: list of lists of tags, excluding START_TAG and END_TAG
        """
        batch_size, max_seq_len, tagset_size = lstm_feats.size()

        # Initialize variables
        backpointers = []

        # Initialize Viterbi variables
        init_vvars = torch.full(
            (batch_size, tagset_size), -10000.0, device=lstm_feats.device
        )
        init_vvars[:, 0] = 0  # START_TAG has all the score

        # Forward pass: compute the best score for each tag
        forward_var = init_vvars

        for t in range(max_seq_len):
            bptrs_t = []  # Holds the backpointers for this step
            viterbivars_t = []  # Holds the Viterbi variables for this step

            # Add the emission scores for each tag
            next_tag_var = (
                forward_var.unsqueeze(1) + self.transitions
            )  # (batch_size, tagset_size, tagset_size)

            # Find the best score and backpointer for each tag
            bptrs_t = torch.max(next_tag_var, dim=2)[1]  # (batch_size, tagset_size)
            viterbivars_t = torch.max(next_tag_var, dim=2)[
                0
            ]  # (batch_size, tagset_size)

            # Add emission scores
            emit_scores = lstm_feats[:, t]  # (batch_size, tagset_size)
            forward_var = viterbivars_t + emit_scores

            # Apply mask - convert bool mask to float for arithmetic operations
            mask_t = (
                mask[:, t].unsqueeze(1).float()
            )  # Convert to float for multiplication
            forward_var = mask_t * forward_var + (1 - mask_t) * init_vvars

            backpointers.append(bptrs_t)

        # Find the best final score
        terminal_var = forward_var
        best_tag_scores, best_tags = torch.max(terminal_var, dim=1)

        # Follow the backpointers to decode the best path
        best_paths = [[best_tag.item()] for best_tag in best_tags]

        # Follow the backpointers to decode the best path
        for bptrs_t in reversed(backpointers):
            for i, path in enumerate(best_paths):
                best_tag = path[0]
                best_tag = bptrs_t[i, best_tag].item()
                path.insert(0, best_tag)

        # Remove the START_TAG
        for path in best_paths:
            del path[0]

        return best_paths

    def _viterbi_decode(self, lstm_feats, mask):
        """
        Find the most likely tag sequence using Viterbi algorithm
        Args:
            lstm_feats: (batch_size, max_seq_len, tagset_size)
            mask: (batch_size, max_seq_len)
        Returns:
            best_scores: (batch_size) Score of the best path
            best_paths: list of lists of tags, excluding START_TAG and END_TAG
        """
        best_paths = self._viterbi_decode_batched(lstm_feats, mask)

        # Get the score of the best path
        batch_size, max_seq_len, tagset_size = lstm_feats.size()
        best_scores = torch.zeros(batch_size, device=lstm_feats.device)

        # Convert paths to tensors for scoring
        for i, path in enumerate(best_paths):
            path_tensor = torch.LongTensor(path).to(lstm_feats.device)
            # Ensure path_tensor has the right size
            if path_tensor.size(0) < max_seq_len:
                padded_path = torch.zeros(
                    max_seq_len, dtype=torch.long, device=lstm_feats.device
                )
                padded_path[: path_tensor.size(0)] = path_tensor
                path_tensor = padded_path

            # Calculate score for the path
            score = self._score_sentence(
                lstm_feats[i : i + 1], path_tensor.unsqueeze(0), mask[i : i + 1]
            )
            best_scores[i] = score

        return best_scores, best_paths
