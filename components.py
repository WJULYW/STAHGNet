import torch
import torch.nn.functional as F
import torch.nn as nn


class AttLayer(nn.Module):

    def __init__(self, in_dim):
        super(AttLayer, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x, y):
        dim = len(x.size())
        if dim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif dim == 3:
            x = x.unsqueeze(-1)
        else:
            pass

        batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(y).view(batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(y).view(batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, width, height)

        out = self.gamma * out + y
        if dim == 2:
            out = out.squeeze(-1).squeeze(-1)
        elif dim == 3:
            out = out.squeeze(-1)
        else:
            pass
        return out


class SelfAtt(nn.Module):
    def __init__(self, out_channels, use_bias=False, reduction=16):
        super(SelfAtt, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, 1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1)
        return x * y.expand_as(x)




class AVWGCN(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.cheb_polynomials = cheb_polynomials
        self.L_tilde = L_tilde
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        # for existing graph convolution
        # self.init_gconv = nn.Conv1d(dim_in, dim_out, kernel_size=5, padding=0)
        self.init_gconv = nn.Linear(dim_in, dim_out)
        self.gconv = nn.Linear(dim_out * cheb_k, dim_out)
        self.dy_gate1 = SelfAtt(dim_out)
        self.dy_gate2 = SelfAtt(dim_out)

    def forward(self, x, node_embeddings, L_tilde_learned):
        b, n, _ = x.shape
        node_num = node_embeddings.shape[0]

        support_set = [torch.eye(node_num).to(L_tilde_learned.device), L_tilde_learned]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * L_tilde_learned, support_set[-1]) - support_set[-2])

        # 1) convolution with learned graph convolution (implicit knowledge)
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv0 = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out

        # 2) convolution with existing graph (explicit knowledge)
        graph_supports = torch.stack(self.cheb_polynomials, dim=0)  # [k, n, m]
        x = self.init_gconv(x)
        x_g1 = torch.einsum("knm,bmc->bknc", graph_supports, x)
        x_g1 = x_g1.permute(0, 2, 1, 3).reshape(b, n, -1)  # B, N, cheb_k, dim_in
        x_gconv1 = self.gconv(x_g1)

        # 3) fusion of explit knowledge and implicit knowledge
        x_gconv = self.dy_gate1(F.leaky_relu(x_gconv0).transpose(1, 2)) + self.dy_gate2(
            F.leaky_relu(x_gconv1).transpose(1, 2))
        # x_gconv = F.leaky_relu(x_gconv0) + F.leaky_relu(x_gconv1)

        return x_gconv.transpose(1, 2)


class RGCNCell(nn.Module):
    def __init__(self, polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(RGCNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(polynomials, L_tilde, dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(polynomials, L_tilde, dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings, learned_tilde):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, learned_tilde))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, learned_tilde))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)




class RGCN(nn.Module):
    def __init__(self, polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(RGCN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self._cells = nn.ModuleList()
        self._cells.append(RGCNCell(polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self._cells.append(RGCNCell(polynomials, L_tilde, node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, learned_tilde):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self._cells[i](current_inputs[:, t, :, :], state, node_embeddings, learned_tilde)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self._cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)





