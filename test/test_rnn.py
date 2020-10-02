import torch
from torch.nn.modules.rnn import LSTMCell, LSTM

from models._modules.rnn_q import LSTMCellQ


def isequal(x, y):
    if (x - y).abs().max() < 1e-4:
        return True
    return False


def test_LSTM(hidden_size=20, input_size=10, batch_size=3, sequence_size=7):
    weight_ih = torch.randn(4 * hidden_size, input_size)
    weight_hh = torch.randn(4 * hidden_size, hidden_size)
    bias_ih = torch.randn(4 * hidden_size)
    bias_hh = torch.randn(4 * hidden_size)

    rnn = LSTM(input_size, hidden_size)
    cell = LSTMCell(input_size, hidden_size)
    rnn.state_dict()['weight_ih_l0'].copy_(weight_ih)
    rnn.state_dict()['weight_hh_l0'].copy_(weight_hh)
    rnn.state_dict()['bias_ih_l0'].copy_(bias_ih)
    rnn.state_dict()['bias_hh_l0'].copy_(bias_hh)

    cell.state_dict()['weight_ih'].copy_(weight_ih)
    cell.state_dict()['weight_hh'].copy_(weight_hh)
    cell.state_dict()['bias_ih'].copy_(bias_ih)
    cell.state_dict()['bias_hh'].copy_(bias_hh)

    x = torch.randn(sequence_size, batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    cx = torch.randn(batch_size, hidden_size)

    assert isequal(rnn.state_dict()['weight_ih_l0'], cell.state_dict()['weight_ih'])
    hx1, cx1 = rnn(x, (hx.unsqueeze(0), cx.unsqueeze(0)))
    hx2, cx2 = hx, cx
    for i in range(sequence_size):
        hx2, cx2 = cell(x[i], (hx2, cx2))
    assert isequal(hx1[-1].squeeze(0), hx2)
    assert isequal(cx1[-1].squeeze(0), cx2)


def test_LSTMCell(hidden_size=64, input_size=74, batch_size=128, sequence_size=13):
    weight_ih = torch.randn(4 * hidden_size, input_size)
    weight_hh = torch.randn(4 * hidden_size, hidden_size)
    bias_ih = torch.randn(4 * hidden_size)
    bias_hh = torch.randn(4 * hidden_size)

    rnn = LSTMCell(input_size, hidden_size)
    rnnQ = LSTMCellQ(input_size, hidden_size)

    rnn.state_dict()['weight_ih'].copy_(weight_ih)
    rnn.state_dict()['weight_hh'].copy_(weight_hh)
    rnn.state_dict()['bias_ih'].copy_(bias_ih)
    rnn.state_dict()['bias_hh'].copy_(bias_hh)

    rnnQ.state_dict()['weight_ih'].copy_(weight_ih)
    rnnQ.state_dict()['weight_hh'].copy_(weight_hh)
    rnnQ.state_dict()['bias_ih'].copy_(bias_ih)
    rnnQ.state_dict()['bias_hh'].copy_(bias_hh)

    input = torch.randn(sequence_size, batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    cx = torch.randn(batch_size, hidden_size)

    assert isequal(rnn.state_dict()['weight_ih'], rnnQ.state_dict()['weight_ih'])
    hx1, cx1 = rnn(input[0], (hx, cx))
    hx2, cx2 = rnnQ(input[0], (hx, cx))
    assert isequal(hx1, hx2)
    assert isequal(cx1, cx2)


if __name__ == '__main__':
    test_LSTMCell()
    test_LSTM()
