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

