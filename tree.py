from copy import deepcopy
import operator
import random
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np

class Operator:
    def __init__(self, op_symbol, op_function, input_count):
        self.op_symbol = op_symbol
        self.op_function = op_function
        self.input_count = input_count

    def __call__(self, *operands):
        return self.op_function(*operands)

    def __str__(self):
        return self.op_symbol

    def __repr__(self):
        return self.op_symbol

class Tree:
    def __init__(self, depth_limit, operator_set: list[Operator], input_vars, constant_vals):
        self.depth_limit = depth_limit
        self.operator_set: list[Operator] = operator_set
        self.input_vars = input_vars
        self.constant_vals = constant_vals
        self.tree_head = None
        self.branches = []

    @property
    def is_terminal(self):
        return not self.branches

    def _generate_sub_expression(self, current_depth):
        if current_depth >= self.depth_limit or random.random() < current_depth / self.depth_limit:
            # Generate a terminal node (input variable or constant)
            return Tree(self.depth_limit, self.operator_set, self.input_vars, self.constant_vals), random.choice(
                self.input_vars + self.constant_vals)
        else:
            # Generate an operator node
            chosen_operator: Operator = random.choice(self.operator_set)
            param_count = chosen_operator.input_count
            expression = Tree(self.depth_limit, self.operator_set, self.input_vars, self.constant_vals)
            expression.tree_head = chosen_operator
            for _ in range(param_count):
                sub_expression, sub_expression_val = self._generate_sub_expression(current_depth + 1)
                sub_expression.tree_head = sub_expression_val
                expression.branches.append(sub_expression)

                assert sub_expression.tree_head is not None
            return expression, chosen_operator

    @staticmethod
    def construct_expression(operator_set, input_vars, constant_vals, depth_limit=10):
        """Construct a random expression tree with a specified depth limit."""
        expression = Tree(depth_limit, operator_set, input_vars, constant_vals)
        expression, _ = expression._generate_sub_expression(0)
        return expression

    def _single_point_mutation(self):
        all_nodes = self._gather_all_nodes()
        target_node, _ = random.choice(all_nodes)
        if not target_node.branches:
            # It's a terminal node
            if random.random() < 0.8:
                target_node.tree_head = random.choice(self.input_vars + self.constant_vals)
            else:
                target_node.tree_head = random.choice(self.constant_vals)
        else:
            # It's an operator node
            target_node.tree_head = random.choice(
                [op for op in self.operator_set if op.input_count == target_node.tree_head.input_count])

        return target_node

    def _sub_expression_mutation(self):
        """Replace a random sub-expression with another randomly generated sub-expression."""

        all_nodes = self._gather_all_nodes()
        sub_expression, current_depth = random.choice(all_nodes)
        branch_count = len(sub_expression.branches)

        if branch_count > 1:
            sub_expression.branches[random.randint(0, branch_count - 1)], _ = self._generate_sub_expression(
                current_depth + 1)
        elif branch_count == 1:
            sub_expression.branches[0], _ = self._generate_sub_expression(current_depth + 1)

    def _permutation_mutation(self):
        """Shuffle the operands of a randomly selected sub-expression"""

        all_nodes = self._gather_all_nodes()
        target_node, _ = random.choice(all_nodes)
        random.shuffle(target_node.branches)

    def _subtree_replacement_mutation(self):
        """Replace the entire expression with a randomly selected sub-expression."""
        all_nodes = self._gather_all_nodes()
        if len(all_nodes) > 1:
            sub_expression, _ = random.choice(all_nodes[1:])
            self.tree_head = sub_expression.tree_head
            self.branches = sub_expression.branches

        assert self._gather_all_nodes()

    def _expansion_mutation(self):
        """Expand a randomly chosen terminal node into an operator with operands."""

        terminal_nodes = self._gather_all_terminal_nodes()
        if terminal_nodes:
            target_node = random.choice(terminal_nodes)
            new_sub_expression, _ = self._generate_sub_expression(0)
            target_node.tree_head = new_sub_expression.tree_head
            target_node.branches = new_sub_expression.branches

    def _collapsing_mutation(self):
        """Collapse a randomly chosen sub-expression into a terminal node."""
        all_nodes = self._gather_all_nodes()
        if len(all_nodes) > 1:
            sub_expression, _ = random.choice(all_nodes[1:])
            terminal_val = random.choice(self.input_vars + self.constant_vals)
            sub_expression.tree_head = terminal_val
            sub_expression.branches = []

    @staticmethod
    def _apply_mutation(expression: "Tree") -> "Tree":
        """Perform a random mutation on the expression with a given probability."""
        mutated_expression: Tree = deepcopy(expression)
        mutated_expression._single_point_mutation()

        return mutated_expression

    @staticmethod
    def _recombine_expressions(expression1: "Tree", expression2: "Tree") -> "Tree":
        """Recombine two expressions by exchanging a random sub-expression from expression1 with one from expression2."""
        recombined_expression = deepcopy(expression1)
        nodes_expression1 = recombined_expression._gather_all_nodes()
        nodes_expression2 = expression2._gather_all_nodes()

        if len(nodes_expression1) > 1 and len(nodes_expression2) > 1:
            sub_expression1, _ = random.choice(nodes_expression1[1:])
            sub_expression2, _ = random.choice(nodes_expression2[1:])
            sub_expression1.tree_head = sub_expression2.tree_head
            sub_expression1.branches = deepcopy(sub_expression2.branches)

        return recombined_expression

    def render_expression(self):
        """Render the expression tree using networkx in a tree-like structure."""
        graph = nx.DiGraph()
        node_labels = {}
        node_counter = 0

        def add_graph_nodes(node, parent_node=None):
            nonlocal node_counter
            current_node_id = node_counter
            node_counter += 1
            if parent_node is not None:
                graph.add_edge(parent_node, current_node_id)
            node_labels[current_node_id] = str(node.tree_head.op_symbol) if node.branches else (
                str(f"{node.tree_head:.3f}") if isinstance(node.tree_head, float) else str(node.tree_head))
            for branch in node.branches:
                add_graph_nodes(branch, current_node_id)

        def calculate_tree_layout(graph: nx.DiGraph, root_node, horizontal_spacing=1., level_spacing=0.2,
                                  vertical_position=0, horizontal_center=0.5):
            """
            If there is a cycle, then this will produce a hierarchy, but nodes will be repeated.
                graph: the graph (must be a tree)
                root_node: the root node of current branch
                horizontal_spacing: horizontal space allocated for this branch - avoids overlap with other branches
                level_spacing: gap between levels of hierarchy
                vertical_position: vertical location of root
                horizontal_center: horizontal location of root
            """
            node_positions = {root_node: (horizontal_center, vertical_position)}
            neighbor_nodes = list(graph.neighbors(root_node))
            if len(neighbor_nodes) != 0:
                horizontal_delta = horizontal_spacing / len(neighbor_nodes)
                next_horizontal = horizontal_center - horizontal_spacing / 2 - horizontal_delta / 2
                for neighbor in neighbor_nodes:
                    next_horizontal += horizontal_delta
                    node_positions.update(calculate_tree_layout(graph, neighbor, horizontal_spacing=horizontal_delta,
                                                               level_spacing=level_spacing,
                                                               vertical_position=vertical_position - level_spacing,
                                                               horizontal_center=next_horizontal))
            return node_positions

        plt.figure(figsize=(14, 8))

        add_graph_nodes(self)

        if not graph.nodes:
            graph.add_node(0)

        node_positions = calculate_tree_layout(graph, min(graph.nodes))
        nx.draw(graph, node_positions, with_labels=True, labels=node_labels, node_size=400, node_color='lightblue',
                font_size=10)

    def _gather_all_nodes(self):
        """Helper function to collect all nodes in the expression."""
        collected_nodes = []
        node_stack = [(self, 0)]

        while node_stack:
            current_node, current_depth = node_stack.pop()
            collected_nodes.append((current_node, current_depth))
            child_nodes = list(zip(current_node.branches, [current_depth + 1] * len(current_node.branches)))
            node_stack.extend(child_nodes)

        return collected_nodes

    def _gather_all_terminal_nodes(self):
        """Helper function to collect all terminal nodes in the expression."""
        terminal_nodes = []

        def traverse_tree(node):
            if not node.branches:
                terminal_nodes.append(node)
            for branch in node.branches:
                traverse_tree(branch)

        traverse_tree(self)
        return terminal_nodes

    @staticmethod
    def calculate_expression(expression: "Tree", input_data: np.array) -> np.array:
        """Calculate the value of the expression for a given input."""

        def calculate_node_value(node, data_row):
            if not node.branches:
                # It's a terminal node
                if node.tree_head in expression.input_vars:
                    return data_row[expression.input_vars.index(node.tree_head)]
                else:
                    return node.tree_head
            else:
                # It's an operator node
                operand_values = [calculate_node_value(branch, data_row) for branch in node.branches]
                return node.tree_head(*operand_values)

        return np.array([calculate_node_value(expression, data_row) for data_row in input_data.T])

    @staticmethod
    def calculate_mean_squared_error(expression: "Tree", input_data: np.array, target_values: np.array) -> float:
        """Calculate the mean squared error of the expression for given input data and target values."""
        predicted_values = Tree.calculate_expression(expression, input_data)
        return (sum(np.clip((a - b), -1e150, 1e150) ** 2 for a, b in zip(target_values, predicted_values)) / len(
            target_values)).astype(float)

    def __repr__(self):
        """Retrieve the symbolic representation of the expression by performing an in-order traversal."""

        def traverse_tree(node):
            if not node.branches:
                return f"{node.tree_head:.3f}" if isinstance(node.tree_head, float) else str(node.tree_head)
            left_operand = traverse_tree(node.branches[0])
            right_operand = traverse_tree(node.branches[1]) if len(node.branches) > 1 else ""
            return f"({left_operand} {node.tree_head.op_symbol} {right_operand})"

        return traverse_tree(self)

if __name__ == "__main__":

    # Define some example operators and variables
    common_operators = [
        Operator('+', operator.add, 2),
        Operator('-', operator.sub, 2),
        Operator('*', operator.mul, 2),
        Operator('/', operator.truediv, 2),
        Operator('^', operator.pow, 2),
        Operator('|.|', operator.abs, 1),
        Operator('-', operator.neg, 1)
    ]

    extended_operators = [
        Operator("|.|", np.abs, 1),
        Operator("-", np.negative, 1),
        Operator("+", np.add, 2),
        Operator("-", np.subtract, 2),
        Operator("*", np.multiply, 2),
        Operator("/", np.divide, 2),
        Operator("^", np.power, 2),
        Operator("sign", np.sign, 1),
        Operator("exp", np.exp, 1),
        Operator("exp2", np.exp2, 1),
        Operator("sqrt", np.sqrt, 1),
        Operator("square", np.square, 1),
        Operator("cbrt", np.cbrt, 1),
        Operator("reciprocal", np.reciprocal, 1),
        Operator("sin", np.sin, 1),
        Operator("cos", np.cos, 1),
        Operator("tan", np.tan, 1),
        Operator("sinh", np.sinh, 1),
        Operator("cosh", np.cosh, 1),
        Operator("tanh", np.tanh, 1)
    ]

    special_operators = [
        Operator("log", np.log, 1),
        Operator("log2", np.log2, 1),
        Operator("log10", np.log10, 1),
        Operator("arcsin", np.arcsin, 1),
        Operator("arccos", np.arccos, 1),
        Operator("arctan", np.arctan, 1)
    ]

    full_operator_set = common_operators + extended_operators

    input_variables = ['x0', 'x1']
    constant_values = [10 * random.random() for _ in range(5)]
    data = np.load("./data/problem_0.npz")

    for i in range(100):
        expression = Tree.construct_expression(depth_limit=5, operator_set=full_operator_set,
                                                         input_vars=input_variables, constant_vals=constant_values)
        print(f"Original Expression: {expression}")
        expression.render_expression()

        X = np.random.rand(2, 3)
        y = np.random.randint(0, 10, 3)

        print(f"Shapes: {X.shape}, {y.shape}")

        print(Tree.calculate_mean_squared_error(expression, X, y))

        expression = Tree._apply_mutation(expression)
        print(f"After Mutation: {expression}")
        expression.render_expression()

        # expression._sub_expression_mutation()
        # print("After Sub-expression Mutation:")
        # expression.render_expression()

        # expression._permutation_mutation()
        # print("After Permutation Mutation:")
        # expression.render_expression()

        # expression._subtree_replacement_mutation()
        # print("After Subtree Replacement Mutation:")
        # expression.render_expression()

        # expression._expansion_mutation()
        # print("After Expansion Mutation:")
        # expression.render_expression()

        # expression._collapsing_mutation()
        # print("After Collapsing Mutation:")
        # expression.render_expression()

        # other_expression = ExpressionTree.construct_expression(depth_limit=10, operator_set=full_operator_set,
        #                                                      input_vars=input_variables,
        #                                                      constant_vals=constant_values)
        # print("Other Individual Expression:")
        # other_expression.render_expression()

        # ExpressionTree._recombine_expressions(expression, other_expression)
        # print("After Sub-expression Exchange:")
        # expression.render_expression()
        # other_expression.render_expression()

        # plt.show()