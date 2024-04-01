import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class VulbertaConfig(PretrainedConfig):

    def __init__(
        self,
        n_classes,
        base_model_output_size,
        dropout=0.1,
        **kwargs,
    ):
        self.n_classes = n_classes
        super().__init__(**kwargs)


class VulbertaVanilla(PreTrainedModel):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.1):
        super().__init__()

        self.num_labels = n_classes
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, n_classes)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        x = outputs[0]
        x = x[:, 0, :]
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        # Below is the standard output from RobertaforSequenceClassifcation head class

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VulbertaExtend(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.1):
        super().__init__()

        self.num_labels = n_classes
        self.base_model = base_model
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        x = outputs[0]
        x = x[:, 0, :]
        x = self.dropout1(x)
        x = nn.functional.selu(self.fc1(x))
        x = self.dropout2(x)
        x = nn.functional.selu(self.fc2(x))
        logits = self.fc3(x)

        # Below is the standard output from RobertaforSequenceClassifcation head class

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VulbertaCNN(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.2):
        super().__init__()

        self.num_labels = n_classes
        self.base_model = base_model
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(300, 128)
        self.fc3 = nn.Linear(128, n_classes)

#        self.conv = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=9)

        self.conv1 = nn.Conv1d(
            in_channels=768, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv1d(
            in_channels=768, out_channels=100, kernel_size=4)
        self.conv3 = nn.Conv1d(
            in_channels=768, out_channels=100, kernel_size=5)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        x = outputs[0]
        # x = x[:, 0, :]
        x = x.permute(0, 2, 1)

        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(x))
        x3 = nn.functional.relu(self.conv3(x))

        x1 = nn.functional.max_pool1d(x1, x1.shape[2])
        x2 = nn.functional.max_pool1d(x2, x2.shape[2])
        x3 = nn.functional.max_pool1d(x3, x3.shape[2])

        x = torch.cat([x1, x2, x3], dim=1)
        x = x.flatten(1)

#         x = nn.functional.relu(self.conv(x))
#         x = nn.functional.max_pool1d(x, 4)
#         x = torch.mean(x, -1)
#         x = self.dropout1(x)

        x = self.fc2(x)
        logits = self.fc3(x)

        # Below is the standard output from RobertaforSequenceClassifcation head class

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VulbertaLSTM(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.2):
        super().__init__()

        self.num_labels = n_classes
        self.base_model = base_model
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(256*2, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.lstm1 = nn.LSTM(input_size=768,
                             hidden_size=256,
                             num_layers=2,
                             batch_first=True,
                             bidirectional=True)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        x = outputs[0]
        # x = x[:, 0, :]
        self.lstm1.flatten_parameters()
        output, (hidden, cell) = self.lstm1(x)
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc2(x))
        logits = self.fc3(x)

        # Below is the standard output from RobertaforSequenceClassifcation head class

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
