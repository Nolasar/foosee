import numpy as np
from src.initializers import GlorotUniform, Zeros
from src.layers.dense import Dense
from src.activations import Relu, Linear, Sigmoid

class MegNet:
    def __init__(
        self,
        node_size: int,
        edge_size: int,
        u_size: int,
        weights_initializer=GlorotUniform,
        bias_inittializer=Zeros,
    ):
        self.weights_initializer = weights_initializer()
        self.bias_inittializer = bias_inittializer()
        self.node_size = node_size
        self.edge_size = edge_size
        self.u_size = u_size

    def compile(self, input_node_shape, input_edge_shape, input_u_shape):
        _, self.node_count, node_features = input_node_shape
        _, self.edge_count, edge_features = input_edge_shape
        _, u_features = input_u_shape
        # ------------------ EDGE ------------------
        self.w_edge = self.weights_initializer(
            shape=(2*node_features + edge_features + u_features, self.edge_size)
        )
        self.b_edge = self.bias_inittializer(shape=(self.edge_size,))
        # ------------------ NODE ------------------
        self.w_node = self.weights_initializer(
            shape=(node_features + self.edge_size + u_features, self.node_size)
        )
        self.b_node = self.bias_inittializer(shape=(self.node_size,))
        # ------------------ U ---------------------
        self.w_u = self.weights_initializer(
            shape=(self.node_size + self.edge_size + u_features, self.u_size)
        )
        self.b_u = self.bias_inittializer(shape=(self.u_size,))
        

    def forward(self, nodes: np.ndarray, edges: np.ndarray, node_idx: np.ndarray, u: np.ndarray):
        """
        nodes:    (batch_size, num_nodes, node_features)
        edges:    (batch_size, num_edges, edge_features)
        node_idx: (batch_size, num_edges, 2) – индексы (v_s, v_r) для каждого ребра
        u:        (batch_size, u_features)
        """
        self.batch_size = nodes.shape[0]
        # 1) Получаем v_s и v_r по индексам
        v_s = nodes[np.arange(self.batch_size)[:, None], node_idx[..., 0]]
        v_r = nodes[np.arange(self.batch_size)[:, None], node_idx[..., 1]]

        # 2) Обновление рёбер: new_edges = φ_e(e_k ⊕ v_sk ⊕ v_rk ⊕ u)
        u_edge = np.repeat(u[:, None, :], self.edge_count, axis=1)
        edge_concat = np.concatenate([edges, v_s, v_r, u_edge], axis=-1)
        new_edges_pre = edge_concat @ self.w_edge + self.b_edge  # (B, E, edge_size)

        new_edges = np.log(1+np.exp(new_edges_pre))

        # 3) Агрегируем (суммируем) новые ребра по получателям:
        receiver_idx = node_idx[..., 1]
        node_count = nodes.shape[1]
        scatter_mask = np.zeros(
            (self.batch_size, node_count, self.edge_count), dtype=bool
        )
        scatter_mask[
            np.arange(self.batch_size)[:, None, None],
            receiver_idx[:, None, :],
            np.arange(self.edge_count)
        ] = True

        # aggregated_edge_features[b,i,f] = sum по e тех new_edges[b,e,f], где r_k = i
        aggregated_edge_features = np.einsum(
            'bef,bie->bif',
            new_edges,
            scatter_mask.astype(int)
        )
        # Усредняем по количеству рёбер на узел
        node_edge_counts = scatter_mask.sum(axis=-1, keepdims=True)
        node_edge_counts[node_edge_counts == 0] = 1  # защита от деления на 0
        averaged_edge_features = aggregated_edge_features / node_edge_counts

        # 4) Обновление узлов: new_nodes = φ_v( v_i ⊕ v̄_i^e ⊕ u )
        u_node = np.repeat(u[:, None, :], node_count, axis=1)
        node_concat = np.concatenate([averaged_edge_features, nodes, u_node], axis=-1)
        new_nodes_pre = node_concat @ self.w_node + self.b_node  # (B, N, node_size)
        new_nodes = np.log(1+np.exp(new_nodes_pre))

        # 5) Обновление глобального признака: new_u = φ_u( mean_edges ⊕ mean_nodes ⊕ u )
        u_edge_mean = new_edges.mean(axis=1)  # (B, edge_size)
        u_node_mean = new_nodes.mean(axis=1)  # (B, node_size)
        u_concat = np.concatenate([u_edge_mean, u_node_mean, u], axis=-1)
        new_u_pre = u_concat @ self.w_u + self.b_u  # (B, u_size)
        new_u = np.log(1+np.exp(new_u_pre))
        # ---------- Сохранение для backward ----------
        self.new_nodes_pre_act = new_nodes_pre
        self.new_edges_pre_act = new_edges_pre
        self.new_u_pre_act = new_u_pre

        self.nodes = nodes
        self.edges = edges
        self.u = u
        self.node_idx = node_idx

        self.v_s = v_s
        self.v_r = v_r
        self.edge_concat = edge_concat
        self.new_edges = new_edges

        self.scatter_mask = scatter_mask
        self.aggregated_edge_features = aggregated_edge_features
        self.node_edge_counts = node_edge_counts
        self.averaged_edge_features = averaged_edge_features

        self.node_concat = node_concat
        self.new_nodes = new_nodes

        self.u_edge_mean = u_edge_mean
        self.u_node_mean = u_node_mean
        self.u_concat = u_concat
        self.new_u = new_u
        # ---------------------------------------------

        return new_nodes, new_edges, node_idx, new_u

    def backward(self,
                 d_new_nodes: np.ndarray,
                 d_new_edges: np.ndarray,
                 d_new_u: np.ndarray
                 ):
        """
        Параметры:
        ----------
        d_new_nodes : (batch_size, node_count, node_size)
            Градиент функции потерь по выходу new_nodes.
        d_new_edges : (batch_size, edge_count, edge_size)
            Градиент функции потерь по выходу new_edges.
        d_new_u     : (batch_size, u_size)
            Градиент функции потерь по выходу new_u.

        Возвращает:
        -----------
        d_nodes : (batch_size, node_count, node_features)
        d_edges : (batch_size, edge_count, edge_features)
        d_u     : (batch_size, u_features)

        А также внутри сохраняет dW и db для оптимизации.
        """

        # --------------------------------------------------
        # 1) ОБРАТНЫЙ ПРОХОД ДЛЯ new_u = u_concat @ w_u + b_u
        # --------------------------------------------------
        # new_u: (B, u_size)
        # u_concat: (B, (edge_size + node_size + u_features))
        # w_u: ((edge_size + node_size + u_features), u_size)
        # b_u: (u_size,)

        # dL/d(u_concat) = dL/d(new_u) @ (w_u^T)
        # dL/d(w_u)      = (u_concat^T) @ (dL/d(new_u))   [суммируя по batch-измерению]
        # dL/d(b_u)      = sum(dL/d(new_u))               [по batch-измерению]
        d_new_u *= 1 / (1 + np.exp(-self.new_u_pre_act))
        d_u_concat = d_new_u @ self.w_u.T  # (B, edge_size+node_size+u_features)

        self.dw_u = self.u_concat.T @ d_new_u  # (B, 1, u_size)
        self.db_u = d_new_u.sum(axis=0)  # (u_size,)

        # print(self.dw_u.shape, self.w_u.shape)
        # print(self.db_u.shape, self.b_u.shape)
        # Расщепляем d_u_concat на три части: d(u_edge_mean), d(u_node_mean), d(u).
        edge_size = self.edge_size
        node_size = self.node_size
        d_u_edge_mean = d_u_concat[..., :edge_size]                # (B, edge_size)
        d_u_node_mean = d_u_concat[..., edge_size:edge_size+node_size]  # (B, node_size)
        d_u_from_u = d_u_concat[..., edge_size+node_size:]         # (B, u_features)

        # print(f'd_u_edge_mean: {d_u_edge_mean.shape}')
        # print(f'd_u_node_mean: {d_u_node_mean.shape}')
        # print(f'd_u_from_u: {d_u_from_u.shape}')

        # -----------------------------------------------------------
        # 2) УЧЁТ СРЕДНЕГО ПО РЁБРАМ И УЗЛАМ ДЛЯ u_edge_mean/u_node_mean
        #    u_edge_mean = mean(new_edges, axis=1)
        #    u_node_mean = mean(new_nodes, axis=1)
        # -----------------------------------------------------------
        # dL/d(new_edges) от этого шага: d_u_edge_mean / d(new_edges) = d_u_edge_mean / E
        # (т.к. mean по оси 1 => каждая компонента умножается на 1 / edge_count)
        # Аналогично для new_nodes.

        B, E, F_e = self.new_edges.shape  # (batch_size, edge_count, edge_size)
        B, N, F_v = self.new_nodes.shape  # (batch_size, node_count, node_size)

        d_new_edges_from_u_mean = (
            d_u_edge_mean[:, None, :] / E  # (B,1,edge_size)
        )  # затем вещаем на каждое ребро
        # broadcasting до (B,E,edge_size)
        d_new_edges_from_u_mean = np.repeat(d_new_edges_from_u_mean, E, axis=1)

        d_new_nodes_from_u_mean = (
            d_u_node_mean[:, None, :] / N  # (B,1,node_size)
        )
        d_new_nodes_from_u_mean = np.repeat(d_new_nodes_from_u_mean, N, axis=1)

        # Суммируем с «внешним» d_new_edges / d_new_nodes, который приходит из функции backward "сверху".
        d_new_edges_total = (d_new_edges + d_new_edges_from_u_mean) * (1 / (1 + np.exp(-self.new_edges_pre_act)))
        d_new_nodes_total = (d_new_nodes + d_new_nodes_from_u_mean) * (1 / (1 + np.exp(-self.new_nodes_pre_act)))

        # --------------------------------------------------
        # 3) ОБРАТНЫЙ ПРОХОД ДЛЯ new_nodes = node_concat @ w_node + b_node
        # --------------------------------------------------
        # new_nodes: (B,N,node_size)
        # node_concat: (B,N,node_features + edge_size + u_features)
        # w_node: ((node_features + edge_size + u_features), node_size)
        # b_node: (node_size,)

        d_node_concat = d_new_nodes_total @ self.w_node.T
        # Градиенты по w_node, b_node:
        # Аналогично выше, суммирование по batch_size и по N (числу узлов):
        self.dw_node = np.sum(self.node_concat.transpose(0,2,1) @ d_new_nodes_total, axis=0)
        self.db_node = d_new_nodes_total.sum(axis=(0, 1))  # скаляр по B,N, остаётся (node_size,)

        # print(self.dw_node.shape, self.w_node.shape)
        # print(self.db_node.shape, self.b_node.shape)
        
        # node_concat = [averaged_edge_features, nodes, u_node]
        nf = self.nodes.shape[-1]    # node_features
        ef = self.edge_size          # уже известный edge_size
        # Итого размер конкатена = nf + ef + uf

        d_averaged_edge_features = d_node_concat[..., :ef]    # (B,N,edge_size)
        d_nodes_from_node_block  = d_node_concat[..., ef:ef+nf]  # (B,N,node_features)
        d_u_node = d_node_concat[..., ef+nf:]  # (B,N,u_features)

        # print(f'd_averaged_edge_features: {d_averaged_edge_features.shape}')
        # print(f'd_nodes_from_node_block: {d_nodes_from_node_block.shape}')
        # print(f'd_u_node: {d_u_node.shape}')

        # print(f'averaged_edge_features: {self.averaged_edge_features.shape}')
        # print(f'nodes_from_node_block: {self.nodes.shape}')

        # Поскольку u_node было повторено на N узлов, складываем градиенты по оси узлов:
        d_u_from_node = d_u_node.sum(axis=1)  # (B, u_features)
        
        # --------------------------------------------------
        # 4) ОБРАТНЫЙ ПРОХОД ЧЕРЕЗ агрегирование в averaged_edge_features
        #    averaged_edge_features = aggregated_edge_features / node_edge_counts
        # --------------------------------------------------
        # d(aggregated_edge_features) = d(averaged_edge_features) / node_edge_counts
        #    (broadcast по соответствующим размерам)
        d_aggregated_edge_features = (
            d_averaged_edge_features * (1.0 / self.node_edge_counts)
        )  # (B,N,edge_size)

        # aggregated_edge_features = einsum('bef,bie->bif', new_edges, scatter_mask)
        # => d new_edges = einsum('bif,bie->bef') (где scatter_mask — 0/1)
        # Формально:
        # d_new_edges[b,e,f] += sum_i( d_aggregated_edge_features[b,i,f] * scatter_mask[b,i,e] )
        d_new_edges_from_node_block = np.einsum(
            'bif,bie->bef',
            d_aggregated_edge_features,
            self.scatter_mask.astype(int)
        )

        # Складываем с уже имеющимся d_new_edges_total
        d_new_edges_total += d_new_edges_from_node_block
        
        # --------------------------------------------------
        # 5) ОБРАТНЫЙ ПРОХОД ДЛЯ new_edges = edge_concat @ w_edge + b_edge
        # --------------------------------------------------
        d_edge_concat = d_new_edges_total @ self.w_edge.T
        # Градиенты по w_edge, b_edge:
        self.dw_edge = np.sum(self.edge_concat.transpose(0,2,1) @ d_new_edges_total, axis=0)
        self.db_edge = d_new_edges_total.sum(axis=(0, 1))  # (edge_size,)

        # print(self.dw_edge.shape, self.w_edge.shape)
        # print(self.db_edge.shape, self.b_edge.shape)
        
        # edge_concat = [edges, v_s, v_r, u_edge]
        ef_in = self.edges.shape[-1]  # edge_features
        nf_in = self.nodes.shape[-1]  # node_features

        # Разбиваем d_edge_concat
        d_edges_from_edge_block = d_edge_concat[..., :ef_in]  # (B,E, edge_features)
        dv_s = d_edge_concat[..., ef_in:ef_in+nf_in]          # (B,E, node_features)
        dv_r = d_edge_concat[..., ef_in+nf_in:ef_in+2*nf_in]  # (B,E, node_features)
        d_u_edge = d_edge_concat[..., ef_in+2*nf_in:]         # (B,E, u_features)

        # --------------------------------------------------
        # 6) УЧЁТ ПОВТОРЕНИЯ u_edge => суммируем градиенты по оси E
        # --------------------------------------------------
        d_u_from_edge = d_u_edge.sum(axis=1)  # (B,u_features)

        # --------------------------------------------------
        # 7) Собираем градиенты по u (из пунктов 1, 3, 6)
        # --------------------------------------------------
        # d_u = d_u_from_u (из new_u-блока)
        #     + d_u_from_node (из new_nodes-блока)
        #     + d_u_from_edge (из new_edges-блока)
        d_u = d_u_from_u + d_u_from_node + d_u_from_edge

        # --------------------------------------------------
        # 8) Градиенты по nodes: нужно аккуратно «раскидать» dv_s, dv_r
        #    потому что v_s, v_r = nodes[b, node_idx[b,e,0/1]]
        # --------------------------------------------------
        B, E, _ = dv_s.shape
        d_nodes = np.zeros_like(self.nodes)  # (B, N, node_features)

        # Индексируем:
        #   v_s[b,e,:] = nodes[b, node_idx[b,e,0], :]
        #   v_r[b,e,:] = nodes[b, node_idx[b,e,1], :]
        # Поэтому:
        #   d_nodes[b,node_idx[b,e,0]] += dv_s[b,e]
        #   d_nodes[b,node_idx[b,e,1]] += dv_r[b,e]

        # Можно сделать циклом или через продвинутое индексирование:
        # Способ через «расщепление»:
        b_ar = np.arange(B)[:, None]      # (B,1)

        # Для v_s:
        d_nodes[b_ar, self.node_idx[..., 0]] += dv_s
        # Для v_r:
        d_nodes[b_ar, self.node_idx[..., 1]] += dv_r

        # Теперь добавляем к этому d_nodes_from_node_block (прямой градиент из node-блока)
        d_nodes += d_nodes_from_node_block

        d_edges = d_edges_from_edge_block
        return d_nodes, d_edges, d_u
    
    # def get_params(self):
    #     return {'weights': self.weights, 'bias': self.bias}

    # def get_grads(self):
    #     return {
    #         'node_weights': self.dweights, 'node_bias': self.dbias,
    #         'edge_weights': self.dweights, 'edge_bias': self.dbias,
    #         'u_weights': self.dweights, 'u_bias': self.dbias,
    #         }

class MegNetBlock:
    def __init__(
            self, 
            hidden_units:int=32,
            megnet_output_sizes:tuple=(64,128,16)
            ):
        self.hidden_units = hidden_units
        self.dense_node = [
            Dense(units=64, activation=Relu()),
            Dense(units=hidden_units, activation=Relu()),
        ]
        self.dense_edge = [
            Dense(units=64, activation=Relu()),
            Dense(units=hidden_units, activation=Relu()),            
        ]
        self.dense_u = [
            Dense(units=64, activation=Relu()),
            Dense(units=hidden_units, activation=Relu()),              
        ]
        self.node_size, self.edge_size, self.u_size = megnet_output_sizes
        self.megnet = MegNet(node_size=self.node_size, edge_size=self.edge_size, u_size=self.u_size)
        

    def compile(self, batch_size, dense_input, megnet_counts):
        self.batch_size = batch_size 
        self.node_count, self.edge_count = megnet_counts
        self.node_features, self.edge_features, self.u_features = dense_input

        # for node, edge, u in zip(self.dense_node, self.dense_edge, self.dense_u):
        #     node.compile(self.node_features)
        #     edge.compile(self.edge_features)
        #     u.compile(self.u_features)
        # TODO REWRITE MORE BEAUTIFY
        self.dense_node[0].compile(self.node_features)
        self.dense_edge[0].compile(self.edge_features)
        self.dense_u[0].compile(self.u_features)

        self.dense_node[1].compile(64)
        self.dense_edge[1].compile(64)
        self.dense_u[1].compile(64)

        self.megnet.compile(
            input_node_shape=(self.batch_size, self.node_count,  self.hidden_units), 
            input_edge_shape=(self.batch_size,  self.edge_count, self.hidden_units), 
            input_u_shape=(self.batch_size,self.hidden_units)
        )
        return batch_size, (self.node_size, self.edge_size, self.u_size), megnet_counts
    
    def forward(self, nodes: np.ndarray, edges: np.ndarray, u: np.ndarray, node_idx: np.ndarray):
        _nodes = nodes
        _edges = edges 
        _u = u
        for node_layer, edge_layer, u_layer in zip(self.dense_node, self.dense_edge, self.dense_u):
            _nodes = node_layer.forward(_nodes)
            _edges = edge_layer.forward(_edges)
            _u = u_layer.forward(_u)

        _nodes, _edges, _, _u = self.megnet.forward(_nodes, _edges, node_idx, _u)
        return _nodes, _edges, _u, node_idx

    def backward(self, nodes_grad: np.ndarray, edges_grad: np.ndarray, u_grad: np.ndarray, node_idx):
        _dnodes, _dedge, _du = self.megnet.backward(nodes_grad, edges_grad, u_grad)

        for node_layer, edge_layer, u_layer in zip(
            reversed(self.dense_node), reversed(self.dense_edge), reversed(self.dense_u)
            ):
            _dnodes = node_layer.backward(_dnodes)
            _dedge = edge_layer.backward(_dedge)
            _du = u_layer.backward(_du)

        return _dnodes, _dedge, _du, node_idx

    def get_params(self):
        return {
                'node_dense0_weigths': self.dense_node[0].weights,'node_dense0_bias': self.dense_node[0].bias,
                'node_dense1_weigths': self.dense_node[1].weights,'node_dense1_bias': self.dense_node[1].bias,
                'edge_dense0_weigths': self.dense_edge[0].weights,'edge_dense0_bias': self.dense_edge[0].bias,
                'edge_dense1_weigths': self.dense_edge[1].weights,'edge_dense1_bias': self.dense_edge[1].bias,
                'u_dense0_weigths': self.dense_u[0].weights,'u_dense0_bias': self.dense_u[0].bias,
                'u_dense1_weigths': self.dense_u[1].weights,'u_dense1_bias': self.dense_u[1].bias,
                'node_weights': self.megnet.w_node, 'node_bias': self.megnet.b_node,
                'edge_weights': self.megnet.w_edge, 'edge_bias': self.megnet.b_edge,
                'u_weights': self.megnet.w_u, 'u_bias': self.megnet.b_u,
            }

    def get_grads(self):
        return {
                'node_dense0_weigths': self.dense_node[0].dweights, 'node_dense0_bias': self.dense_node[0].dbias,
                'node_dense1_weigths': self.dense_node[1].dweights, 'node_dense1_bias': self.dense_node[1].dbias,
                'edge_dense0_weigths': self.dense_edge[0].dweights, 'edge_dense0_bias': self.dense_edge[0].dbias,
                'edge_dense1_weigths': self.dense_edge[1].dweights, 'edge_dense1_bias': self.dense_edge[1].dbias,
                'u_dense0_weigths': self.dense_u[0].dweights, 'u_dense0_bias': self.dense_u[0].dbias,
                'u_dense1_weigths': self.dense_u[1].dweights, 'u_dense1_bias': self.dense_u[1].dbias,
                'node_weights': self.megnet.dw_node, 'node_bias': self.megnet.db_node,
                'edge_weights': self.megnet.dw_edge, 'edge_bias': self.megnet.db_edge,
                'u_weights': self.megnet.dw_u, 'u_bias': self.megnet.db_u,
            }
    

class Flatten:
    def compile(self, *input_shape):
        b, f, n = input_shape
        self.output_dim = f[0] * n[0] + f[1] * n[1] + f[2]
        return self.output_dim 

    def forward(self, inputs):
        nodes, edges, u, node_idx = inputs

        B = nodes.shape[0]
        # Запоминаем "формы" для backward
        self._shape_nodes = nodes.shape  # (B, N, node_dim)
        self._shape_edges = edges.shape  # (B, E, edge_dim)
        self._shape_u     = u.shape      # (B, u_dim)

        # "Разворачиваем" по осям
        nodes_flat = nodes.reshape(B, -1)
        edges_flat = edges.reshape(B, -1)
        u_flat     = u.reshape(B, -1)

        # Объединяем всё в один вектор
        out = np.concatenate([nodes_flat, edges_flat, u_flat], axis=-1)

        # Сохраняем выходную форму (B, total_dim):
        self.output_dim = out.shape[-1]
        self._node_idx  = node_idx  # Если хотим в backward вернуть
        return out

    def backward(self, d_out):
        B = self._shape_nodes[0]

        n_nodes = self._shape_nodes[1] * self._shape_nodes[2]  # N * node_dim
        n_edges = self._shape_edges[1] * self._shape_edges[2]  # E * edge_dim
        n_u     = self._shape_u[1]                             # u_dim

        # Разбиваем d_out
        d_nodes_flat = d_out[:, :n_nodes]
        d_edges_flat = d_out[:, n_nodes:n_nodes + n_edges]
        d_u_flat     = d_out[:, n_nodes + n_edges : n_nodes + n_edges + n_u]

        # Возвращаем в исходные формы
        d_nodes = d_nodes_flat.reshape(self._shape_nodes)
        d_edges = d_edges_flat.reshape(self._shape_edges)
        d_u     = d_u_flat.reshape(self._shape_u)

        return (d_nodes, d_edges, d_u, self._node_idx)
    
    def get_params(self):
        return {}

    def get_grads(self):
        return {}
