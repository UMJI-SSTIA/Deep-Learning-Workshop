from manim import *


class RNNVisualization(Scene):
    def construct(self):
        # Title
        title = Text("Recurrent Neural Network (RNN)", font_size=40)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # --- Rolled View ---
        rolled_title = Text("Rolled View", font_size=24).next_to(title, DOWN, buff=0.5)
        self.play(Write(rolled_title))

        # Input x
        input_label = MathTex(r"x_t", font_size=36)
        input_rect = Rectangle(width=0.8, height=0.6, color=BLUE, fill_opacity=0.5).add(
            input_label
        )
        input_rect.shift(LEFT * 2.5)

        # Output y
        output_label = MathTex(r"y_t", font_size=36)
        output_rect = Rectangle(
            width=0.8, height=0.6, color=GREEN, fill_opacity=0.5
        ).add(output_label)
        output_rect.shift(RIGHT * 2.5)

        # Recurrent Cell h
        cell_label = MathTex(r"h_t", font_size=36)
        cell_circle = Circle(radius=0.7, color=ORANGE, fill_opacity=0.5).add(cell_label)
        cell_circle.shift(UP * 1)

        # Self-loop arrow (recurrent connection)
        loop_arrow = ArcBetweenPoints(
            cell_circle.get_right(), cell_circle.get_top(), radius=0.8
        )
        loop_arrow.add_tip()
        loop_label = MathTex(r"W_{hh}", font_size=24).next_to(loop_arrow, UR, buff=0.1)
        loop_group = VGroup(loop_arrow, loop_label)

        # Arrows connecting input to cell and cell to output
        in_cell_arrow = Arrow(input_rect.get_top(), cell_circle.get_left(), buff=0.1)
        cell_out_arrow = Arrow(cell_circle.get_right(), output_rect.get_top(), buff=0.1)

        # Weight labels for arrows
        weight_xh_label = MathTex(r"W_{hx}", font_size=24).next_to(
            in_cell_arrow, UP, buff=0.1
        )
        weight_hy_label = MathTex(r"W_{yh}", font_size=24).next_to(
            cell_out_arrow, UP, buff=0.1
        )

        rolled_view = VGroup(
            input_rect,
            in_cell_arrow,
            cell_circle,
            loop_group,
            cell_out_arrow,
            output_rect,
            weight_xh_label,
            weight_hy_label,
        )

        self.play(FadeIn(rolled_view))
        self.wait(2)
        self.play(FadeOut(rolled_title), FadeOut(rolled_view), FadeOut(title))

        # --- Unrolled View ---
        unrolled_title = Text("Unrolled View (Through Time)", font_size=24).next_to(
            title, DOWN, buff=0.5
        )
        self.play(Write(unrolled_title))
        self.play(unrolled_title.animate.to_edge(UP))

        time_steps = [r"t-1", r"t", r"t+1"]
        cell_spacing = 3

        cells = []
        inputs = []
        outputs = []
        h_arrows = []  # Arrows between hidden states
        x_arrows = []  # Arrows from inputs to hidden states
        y_arrows = []  # Arrows from hidden states to outputs
        w_xh_labels = []
        w_yh_labels = []
        w_hh_labels = []

        # Create elements for each time step
        for i, t_label in enumerate(time_steps):
            x_offset = (i - 1) * cell_spacing

            # Input x_t
            input_rect_unrolled = Rectangle(
                width=0.8, height=0.6, color=BLUE, fill_opacity=0.5
            ).shift(DOWN * 1.5 + RIGHT * x_offset)
            input_label_unrolled = MathTex(
                r"x_{" + t_label + r"}", font_size=36
            ).move_to(input_rect_unrolled.get_center())
            inputs.append(VGroup(input_rect_unrolled, input_label_unrolled))

            # Output y_t
            output_rect_unrolled = Rectangle(
                width=0.8, height=0.6, color=GREEN, fill_opacity=0.5
            ).shift(UP * 2.5 + RIGHT * x_offset)
            output_label_unrolled = MathTex(
                r"y_{" + t_label + r"}", font_size=36
            ).move_to(output_rect_unrolled.get_center())
            outputs.append(VGroup(output_rect_unrolled, output_label_unrolled))

            # Hidden state h_t
            cell_circle_unrolled = Circle(
                radius=0.7, color=ORANGE, fill_opacity=0.5
            ).shift(UP * 0.5 + RIGHT * x_offset)
            cell_label_unrolled = MathTex(
                r"h_{" + t_label + r"}", font_size=36
            ).move_to(cell_circle_unrolled.get_center())
            cells.append(VGroup(cell_circle_unrolled, cell_label_unrolled))

            # Arrow input -> cell
            x_arrow_unrolled = Arrow(
                input_rect_unrolled.get_top(),
                cell_circle_unrolled.get_bottom(),
                buff=0.1,
            )
            x_arrows.append(x_arrow_unrolled)
            w_xh_labels.append(
                MathTex(r"W_{hx}", font_size=20, color=BLUE).next_to(
                    x_arrow_unrolled, LEFT, buff=0.1
                )
            )

            # Arrow cell -> output
            y_arrow_unrolled = Arrow(
                cell_circle_unrolled.get_top(),
                output_rect_unrolled.get_bottom(),
                buff=0.1,
            )
            y_arrows.append(y_arrow_unrolled)
            w_yh_labels.append(
                MathTex(r"W_{yh}", font_size=20, color=GREEN).next_to(
                    y_arrow_unrolled, LEFT, buff=0.1
                )
            )

            # Arrow cell(t-1) -> cell(t)
            if i > 0:
                h_arrow_unrolled = Arrow(
                    cells[i - 1].get_right(), cells[i].get_left(), buff=0.1
                )
                h_arrows.append(h_arrow_unrolled)
                w_hh_labels.append(
                    MathTex(r"W_{hh}", font_size=20, color=ORANGE).next_to(
                        h_arrow_unrolled, UP, buff=0.1
                    )
                )

        # Initial arrows for h_{t-2} -> h_{t-1} and final h_{t+1} -> h_{t+2}
        h_arrow_start = Arrow(
            cells[0].get_left() + LEFT * 0.5, cells[0].get_left(), buff=0.1
        )
        h_arrows.insert(0, h_arrow_start)
        h_arrow_end = Arrow(
            cells[2].get_right(), cells[2].get_right() + RIGHT * 0.5, buff=0.1
        )
        h_arrows.append(h_arrow_end)

        # Display elements
        self.play(FadeIn(VGroup(*cells), VGroup(*inputs), VGroup(*outputs)))
        self.play(
            FadeIn(
                VGroup(*x_arrows),
                VGroup(*y_arrows),
                VGroup(*h_arrows),
                VGroup(*w_xh_labels),
                VGroup(*w_yh_labels),
                VGroup(*w_hh_labels),
            )
        )
        self.wait(3)

        # Highlight shared weights
        highlight_box_xh = SurroundingRectangle(
            VGroup(*w_xh_labels), color=YELLOW, buff=0.2
        )
        highlight_box_yh = SurroundingRectangle(
            VGroup(*w_yh_labels), color=YELLOW, buff=0.2
        )
        highlight_box_hh = SurroundingRectangle(
            VGroup(*w_hh_labels), color=YELLOW, buff=0.2
        )

        shared_weights_text = Text(
            "Shared Weights", font_size=20, color=YELLOW
        ).next_to(highlight_box_xh, DOWN, buff=0.1)

        self.play(
            Create(highlight_box_xh),
            Create(highlight_box_yh),
            Create(highlight_box_hh),
            Write(shared_weights_text),
        )
        self.wait(2)
        self.play(
            FadeOut(highlight_box_xh),
            FadeOut(highlight_box_yh),
            FadeOut(highlight_box_hh),
            FadeOut(shared_weights_text),
        )

        # --- Equations ---
        eq_title = (
            Text("Key Equations", font_size=24, color=YELLOW)
            .to_edge(LEFT, buff=0.5)
            .shift(UP * 1.5)
        )
        hidden_eq = MathTex(
            r"h_t = \sigma_h(W_{hh} h_{t-1} + W_{hx} x_t + b_h)", font_size=30
        ).next_to(eq_title, DOWN, buff=0.3, aligned_edge=LEFT)
        output_eq = MathTex(r"y_t = \sigma_y(W_{yh} h_t + b_y)", font_size=30).next_to(
            hidden_eq, DOWN, buff=0.3, aligned_edge=LEFT
        )
        eq_group = VGroup(eq_title, hidden_eq, output_eq)

        # Shift unrolled view down to make room for equations
        self.play(
            VGroup(
                unrolled_title,
                VGroup(*cells),
                VGroup(*inputs),
                VGroup(*outputs),
                VGroup(*x_arrows),
                VGroup(*y_arrows),
                VGroup(*h_arrows),
                VGroup(*w_xh_labels),
                VGroup(*w_yh_labels),
                VGroup(*w_hh_labels),
            ).animate.shift(RIGHT * 1.4)
        )

        self.play(FadeIn(eq_group))
        self.wait(3)

        # Highlight connection between equation terms and visualization for time step t
        # (Focusing on h_t equation)
        t_index = 1  # time step t

        h_t_circ = SurroundingRectangle(cells[t_index], color=YELLOW, buff=0.1)
        self.play(Create(h_t_circ))
        self.wait(1)

        h_tm1_circ = SurroundingRectangle(cells[t_index - 1], color=YELLOW, buff=0.1)
        self.play(Create(h_tm1_circ))
        self.wait(1)

        x_t_rect = SurroundingRectangle(inputs[t_index], color=YELLOW, buff=0.1)
        self.play(Create(x_t_rect))
        self.wait(1)

        w_hh_rect = SurroundingRectangle(w_hh_labels[0], color=YELLOW, buff=0.1)
        self.play(Create(w_hh_rect))
        self.wait(1)

        w_xh_rect = SurroundingRectangle(w_xh_labels[t_index], color=YELLOW, buff=0.1)
        self.play(Create(w_xh_rect))
        self.wait(2)

        # Clear highlights
        self.play(
            FadeOut(h_t_circ),
            FadeOut(h_tm1_circ),
            FadeOut(x_t_rect),
            FadeOut(w_hh_rect),
            FadeOut(w_xh_rect),
        )
        self.wait(1)

        # Focus on y_t equation
        y_t_rect = SurroundingRectangle(outputs[t_index], color=YELLOW, buff=0.1)
        self.play(Create(y_t_rect))
        self.wait(1)

        h_t_circ = SurroundingRectangle(cells[t_index], color=YELLOW, buff=0.1)
        self.play(Create(h_t_circ))
        self.wait(1)

        w_yh_rect = SurroundingRectangle(w_yh_labels[t_index], color=YELLOW, buff=0.1)
        self.play(Create(w_yh_rect))
        self.wait(2)

        # Final scene holds
        self.play(FadeOut(y_t_rect), FadeOut(h_t_circ), FadeOut(w_yh_rect))
        self.wait(2)


class VanishingGradientBPTT(Scene):
    def construct(self):
        # Configuration
        sequence_length = 5
        cell_spacing = 2.5
        small_scale = 0.5  # Scale for shrinking gradients

        # Title
        title = Text("Vanishing Gradient in BPTT", font_size=36, color=BLUE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP, buff=0.3))

        # --- Setup the Unrolled Network ---
        hidden_states = VGroup()
        output_losses = VGroup()
        input_x = VGroup()

        # Coordinates for positioning
        y_h = 0
        y_l = 2
        y_x = -2

        for i in range(sequence_length):
            x_offset = (i - (sequence_length - 1) / 2) * cell_spacing

            # Hidden States (h_t)
            h_circ = Circle(radius=0.6, color=ORANGE, fill_opacity=0.3)
            h_label = MathTex(r"h_{" + str(i + 1) + "}", font_size=30).move_to(
                h_circ.get_center()
            )
            h_t = VGroup(h_circ, h_label).move_to(RIGHT * x_offset + UP * y_h)
            hidden_states.add(h_t)

            # Outputs and Losses (L_t)
            l_rect = Rectangle(width=1.0, height=0.7, color=GREEN, fill_opacity=0.3)
            l_label = MathTex(r"L_{" + str(i + 1) + "}", font_size=30).move_to(
                l_rect.get_center()
            )
            l_t = VGroup(l_rect, l_label).move_to(RIGHT * x_offset + UP * y_l)
            output_losses.add(l_t)

            # Inputs (x_t)
            x_rect = Rectangle(width=1.0, height=0.7, color=BLUE, fill_opacity=0.3)
            x_label = MathTex(r"x_{" + str(i + 1) + "}", font_size=30).move_to(
                x_rect.get_center()
            )
            x_t = VGroup(x_rect, x_label).move_to(RIGHT * x_offset + UP * y_x)
            input_x.add(x_t)

        # Recurrent connections (h_t-1 -> h_t)
        h_connections = VGroup(
            *[
                Arrow(
                    hidden_states[i].get_right(),
                    hidden_states[i + 1].get_left(),
                    buff=0.1,
                    color=GRAY,
                )
                for i in range(sequence_length - 1)
            ]
        )

        # Loss connections (h_t -> L_t)
        l_connections = VGroup(
            *[
                Arrow(
                    hidden_states[i].get_top(),
                    output_losses[i].get_bottom(),
                    buff=0.1,
                    color=GRAY,
                )
                for i in range(sequence_length)
            ]
        )

        # Input connections (x_t -> h_t)
        x_connections = VGroup(
            *[
                Arrow(
                    input_x[i].get_top(),
                    hidden_states[i].get_bottom(),
                    buff=0.1,
                    color=GRAY,
                )
                for i in range(sequence_length)
            ]
        )

        # Display the network structure
        network = VGroup(
            hidden_states,
            output_losses,
            input_x,
            h_connections,
            l_connections,
            x_connections,
        )
        self.play(FadeIn(network))
        self.wait(1)

        # --- The Vanishing Derivative Concept ---

        # Show the derivative being repeated
        # 1. Define the components
        deriv_tex = MathTex(
            r"\frac{\partial h_t}{\partial h_{t-1}}",
            r"\approx W_{hh}",
            r"< 1",
            font_size=28,
            color=YELLOW,
        )

        multi_tex = MathTex(
            r"\frac{\partial h_t}{\partial h_k}",
            r" = ",
            r"\prod_{i=k+1}^t",
            r"\frac{\partial h_i}{\partial h_{i-1}}",
            font_size=28,
        )

        # Color the specific term in the product to match deriv_tex
        multi_tex[3].set_color(YELLOW)

        # 2. Group and Position at the bottom
        # Arrange them horizontally with some space between the two formulas
        vanishing_info = VGroup(multi_tex, deriv_tex).arrange(RIGHT, buff=1.5)
        vanishing_info.to_edge(DOWN, buff=0.5)

        # 3. Animation Logic
        # Fade them in or write them at the start of the explanation
        self.play(Write(vanishing_info))
        self.wait(1)

        # --- Visualize Vanishing Gradient Path ---
        path_title = (
            Text("Backpropagating from L5 to h1", font_size=24, color=RED)
            .to_edge(RIGHT, buff=0.5)
            .shift(DOWN * 1.5)
        )
        self.play(Write(path_title))

        # 1. Start Gradient at L5
        grad_label = MathTex(r"\delta L_5", font_size=24, color=RED)
        grad_signal = Circle(radius=0.2, color=RED, fill_opacity=0.8).add(grad_label)
        grad_signal.move_to(output_losses[-1].get_center())
        self.play(FadeIn(grad_signal))
        self.wait(1)

        # 2. Backprop L5 -> h5
        self.play(
            grad_signal.animate.move_to(hidden_states[-1].get_center()).scale(
                0.9
            ),  # Slight shrink for non-recurrent step
            run_time=1,
        )
        self.wait(1)

        # 3. Vanishing Steps: h5 -> h4 -> h3 -> h2 -> h1
        for i in range(sequence_length - 1, 0, -1):
            # The backprop arrow: using a CurvedArrow for a "path" look
            back_arrow = CurvedArrow(
                hidden_states[i].get_left(),
                hidden_states[i - 1].get_right(),
                angle=TAU / 8,
                color=RED,
                stroke_width=2,
            )

            self.play(
                Create(back_arrow),
                grad_signal.animate.move_to(hidden_states[i - 1].get_center()).scale(
                    small_scale
                ),
                run_time=1.2,
            )

            # Briefly highlight the multiplication in the equation
            self.play(Indicate(deriv_tex, color=RED, scale_factor=1.2), run_time=0.4)

            # Indicate the multiplication happening
            indicate_rect = SurroundingRectangle(deriv_tex, color=RED, buff=0.1)
            self.play(Create(indicate_rect), run_time=0.5)
            self.play(FadeOut(indicate_rect), run_time=0.5)
            self.wait(0.5)

        # Final emphasis on the tiny signal
        final_label = Text("Gradient Vanished", font_size=24, color=RED).next_to(
            grad_signal, DOWN, buff=0.2
        )
        self.play(
            Write(final_label), Indicate(grad_signal, color=YELLOW, scale_factor=2)
        )
        self.wait(3)


class LSTMvsRNNGradientSolve(Scene):
    def construct(self):
        # --- Configuration & Styling ---
        seq_len = 4
        spacing = 3.0
        standard_opacity = 0.3

        # Colors
        color_rnn = ORANGE
        color_lstm_cell = GREEN_E  # Dark Green for Cell State
        color_lstm_hidden = GREEN_A  # Light Green for Hidden State
        color_grad = RED
        color_gate = YELLOW

        # Titles
        main_title = Text("Solving Vanishing Gradients: RNN vs LSTM", font_size=36)
        self.add(main_title.to_edge(UP, buff=0.3))
        self.wait(1)

        # Labels for the two rows
        rnn_label = (
            Text("Standard RNN (Vanishing)", font_size=28, color=color_rnn)
            .to_edge(LEFT, buff=0.5)
            .shift(UP * 2.5)
        )
        lstm_label = (
            Text("LSTM (Solving via Gates)", font_size=28, color=color_lstm_cell)
            .to_edge(LEFT, buff=0.5)
            .shift(DOWN * 3.5)
        )
        self.play(Write(rnn_label), Write(lstm_label))
        self.wait(1)

        # --- PART 1: Standard RNN (Top Row) ---
        rnn_states = VGroup()
        rnn_arrows = VGroup()

        y_rnn = 1.5

        for i in range(seq_len):
            x = (i - (seq_len - 1) / 2) * spacing
            # Hidden State h_t
            h_circ = Circle(radius=0.5, color=color_rnn, fill_opacity=standard_opacity)
            h_lab = MathTex(r"h_{" + str(i + 1) + "}", font_size=24).move_to(
                h_circ.get_center()
            )
            h_t = VGroup(h_circ, h_lab).move_to(RIGHT * x + UP * y_rnn)
            rnn_states.add(h_t)

            # Arrows
            if i > 0:
                arrow = Arrow(
                    rnn_states[i - 1].get_right(),
                    rnn_states[i].get_left(),
                    buff=0.1,
                    color=GRAY,
                )
                rnn_arrows.add(arrow)

        self.play(FadeIn(rnn_states), FadeIn(rnn_arrows))

        # RNN Gradient Math (Top-Right)
        rnn_math = (
            MathTex(
                r"\frac{\partial h_t}{\partial h_{t-1}}",
                r" = \text{diag}(\sigma') ",
                r"\cdot W_{hh}",
                font_size=24,
                color=color_rnn,
            )
            .to_edge(RIGHT, buff=0.5)
            .shift(UP * 2.5)
        )

        # Highlight matrix multiplication
        self.play(Write(rnn_math))
        self.play(Indicate(rnn_math[2], color=RED, scale_factor=1.3))
        self.wait(1)

        # --- PART 2: LSTM (Bottom Row) ---
        lstm_cells = VGroup()  # Cell States C_t (Top line)
        lstm_hidden = VGroup()  # Hidden States h_t (Bottom line)
        c_arrows = VGroup()  # Direct Cell State arrows
        h_arrows = VGroup()  # Hidden to Hidden arrows

        y_lstm_c = -1.0  # Cell State Highway
        y_lstm_h = -2.2  # Hidden State

        for i in range(seq_len):
            x = (i - (seq_len - 1) / 2) * spacing

            # Cell State C_t (The Solution Highway)
            c_circ = Circle(
                radius=0.5, color=color_lstm_cell, fill_opacity=standard_opacity
            )
            c_lab = MathTex(r"C_{" + str(i + 1) + "}", font_size=24).move_to(
                c_circ.get_center()
            )
            c_t = VGroup(c_circ, c_lab).move_to(RIGHT * x + UP * y_lstm_c)
            lstm_cells.add(c_t)

            # Hidden State h_t
            h_circ = Circle(
                radius=0.4, color=color_lstm_hidden, fill_opacity=standard_opacity
            )
            h_lab = MathTex(r"h_{" + str(i + 1) + "}", font_size=24).move_to(
                h_circ.get_center()
            )
            h_t = VGroup(h_circ, h_lab).move_to(RIGHT * x + UP * y_lstm_h)
            lstm_hidden.add(h_t)

            # Cell arrows (Direct line)
            if i > 0:
                arrow_c = Arrow(
                    lstm_cells[i - 1].get_right(),
                    lstm_cells[i].get_left(),
                    buff=0.1,
                    color=color_lstm_cell,
                )
                c_arrows.add(arrow_c)

                # Hidden arrows
                arrow_h = Arrow(
                    lstm_hidden[i - 1].get_right(),
                    lstm_hidden[i].get_left(),
                    buff=0.1,
                    color=GRAY_A,
                    stroke_width=1,
                )
                h_arrows.add(arrow_h)

        self.play(
            FadeIn(lstm_cells), FadeIn(c_arrows), FadeIn(lstm_hidden), FadeIn(h_arrows)
        )

        # LSTM Gradient Math (Bottom-Right)
        # We focus on the dominant cell-state term: partial C_t / partial C_t-1
        lstm_math = (
            MathTex(
                r"\frac{\partial C_t}{\partial C_{t-1}}",
                r" = ",
                r"f_t",  # Forget Gate
                r" \odot \dots",
                font_size=24,
                color=color_lstm_cell,
            )
            .to_edge(RIGHT, buff=0.5)
            .shift(DOWN * 0.5)
        )
        lstm_math[2].set_color(color_gate)  # Color the Forget Gate yellow

        self.play(Write(lstm_math))
        self.wait(1)

        # Highlight the Forget Gate as the scalar controller
        gate_info = Text("(Forget Gate)", font_size=18, color=color_gate).next_to(
            lstm_math[2], DOWN, buff=0.1
        )
        self.play(
            Indicate(lstm_math[2], color=color_gate, scale_factor=1.3),
            FadeIn(gate_info),
        )
        self.wait(2)
        self.play(FadeOut(gate_info))

        # --- PART 3: Visualizing the Gradient Contrast ---

        contrast_title = Text(
            "Backpropagating Gradients (Step-by-Step)", font_size=24, color=color_grad
        ).move_to(ORIGIN)
        self.play(FadeOut(main_title), FadeIn(contrast_title))
        self.wait(1)

        # Create Gradient Signals (Circles)
        grad_rnn = Circle(radius=0.3, color=color_grad, fill_opacity=0.9).move_to(
            rnn_states[-1].get_center()
        )
        grad_lstm = Circle(radius=0.3, color=color_grad, fill_opacity=0.9).move_to(
            lstm_cells[-1].get_center()
        )

        self.play(FadeIn(grad_rnn), FadeIn(grad_lstm))
        self.wait(1)

        # -- The Simultaneous Backprop Loop --
        shrink_factor = 0.5  # RNN shrinks by half each step

        # Define the curved backprop paths
        rnn_back_arrows = VGroup(
            *[
                CurvedArrow(
                    rnn_states[i].get_left(),
                    rnn_states[i - 1].get_right(),
                    angle=TAU / 8,
                    color=color_grad,
                )
                for i in range(seq_len - 1, 0, -1)
            ]
        )

        lstm_back_arrows = VGroup(
            *[
                CurvedArrow(
                    lstm_cells[i].get_left(),
                    lstm_cells[i - 1].get_right(),
                    angle=TAU / 8,
                    color=color_grad,
                )
                for i in range(seq_len - 1, 0, -1)
            ]
        )

        for i in range(seq_len - 1):

            # --- RNN Path (Top) ---
            # Indicate multiplication in the math
            self.play(Indicate(rnn_math[2], color=RED), run_time=0.4)

            # Animation: Move, Create Arrow, and Shrink
            self.play(
                Create(rnn_back_arrows[i]),
                grad_rnn.animate.move_to(
                    rnn_states[seq_len - 2 - i].get_center()
                ).scale(
                    shrink_factor
                ),  # Shrink RNN gradient
                run_time=1.2,
            )

            # --- LSTM Path (Bottom) ---
            # Indicate the Forget Gate controller
            self.play(Indicate(lstm_math[2], color=color_gate), run_time=0.4)

            # Scenario: The network wants to remember, so f_t = 1.0 (Gate OPEN)
            gate_status = MathTex(
                r"f_t \approx 1.0", font_size=24, color=color_gate
            ).next_to(lstm_math[2], DOWN, buff=0.1)
            self.play(FadeIn(gate_status), run_time=0.3)

            # Animation: Move, Create Arrow, but keep same scale!
            self.play(
                Create(lstm_back_arrows[i]),
                grad_lstm.animate.move_to(lstm_cells[seq_len - 2 - i].get_center()),
                # NO SCALING applied here
                run_time=1.2,
            )
            self.play(FadeOut(gate_status), run_time=0.3)
            self.wait(0.5)

        # --- Conclusion & Emphasis ---
        self.play(FadeOut(contrast_title))

        conclusion_rnn = Text(
            "Gradient Vanishes", font_size=20, color=color_grad
        ).next_to(grad_rnn, DOWN, buff=0.2)
        conclusion_lstm = Text(
            "Gradient Flow Preserved", font_size=20, color=color_grad
        ).next_to(grad_lstm, DOWN, buff=0.2)

        self.play(Write(conclusion_rnn), Write(conclusion_lstm))
        self.play(
            Indicate(grad_rnn, color=YELLOW, scale_factor=2),
            Indicate(grad_lstm, color=color_lstm_cell, scale_factor=1.5),
        )
        self.wait(4)


class LSTMMathGates(Scene):
    def construct(self):
        # --- Titles and Setup ---
        title = Text("Inside the LSTM Cell: The Mathematics of Gates", font_size=32)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))

        # Colors for different gate types
        color_forget = RED_B
        color_input = GREEN_B
        color_cand = BLUE_B
        color_output = PURPLE_B
        color_state = YELLOW_B

        # --- The Input Vectors ---
        # Representing [h_{t-1}, x_t]
        inputs_box = Rectangle(width=1.5, height=2.5, color=WHITE).shift(LEFT * 5)
        inputs_label = MathTex(
            r"\begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix}", font_size=34
        ).move_to(inputs_box)
        inputs_grp = VGroup(inputs_box, inputs_label)

        self.play(FadeIn(inputs_grp))

        # --- Defining the 4 Gates (The Math) ---
        # We will arrange these in a column
        gate_math = (
            VGroup(
                MathTex(
                    r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)", color=color_forget
                ),
                MathTex(
                    r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)", color=color_input
                ),
                MathTex(
                    r"\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)",
                    color=color_cand,
                ),
                MathTex(
                    r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)", color=color_output
                ),
            )
            .arrange(DOWN, buff=0.6)
            .shift(LEFT * 0.5)
        )

        # --- Animations ---

        # 1. Flowing inputs to gates
        self.play(LaggedStart(*[Write(gate) for gate in gate_math], lag_ratio=0.3))
        self.wait(1)

        # 2. Highlighting Gate Roles
        # Forget Gate
        self.play(Indicate(gate_math[0]))
        forget_desc = Text(
            "Forget: What to drop from memory", font_size=18, color=color_forget
        ).next_to(gate_math[0], UP, buff=0.1)
        self.play(Write(forget_desc))
        self.wait(1)

        # Input + Candidate
        self.play(Indicate(gate_math[1]), Indicate(gate_math[2]))
        input_desc = Text(
            "Update: What new info to add", font_size=18, color=color_input
        ).next_to(gate_math[1], DOWN, buff=0.1)
        self.play(Write(input_desc))
        self.wait(1)

        self.play(Indicate(gate_math[3]))
        output_desc = Text(
            "Output: decide what parts of the cell state to output",
            font_size=18,
            color=color_output,
        ).next_to(gate_math[3], DOWN, buff=0.1)
        self.play(Write(output_desc))
        self.wait(1)

        self.play(FadeOut(*self.mobjects))
        title = Text("Output logic", font_size=26)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        state_math = VGroup(
            MathTex(
                r"C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t",
                color=color_state,
            ),
            MathTex(r"h_t = o_t \odot \tanh(C_t)", color=WHITE),
        ).arrange(DOWN, buff=0.8)
        self.play(Write(state_math))
        self.wait(1)

        # 4. Focusing on the Cell State "Conveyor Belt"
        cell_belt_rect = SurroundingRectangle(
            state_math[0], color=color_state, buff=0.2
        )
        self.play(Create(cell_belt_rect))

        # Adding a comment about the "+" sign - this is the key to solving vanishing gradients
        linear_comment = Text(
            "Linear addition prevents \n gradient vanishing",
            font_size=20,
            color=color_state,
        ).next_to(cell_belt_rect, UP)
        self.play(Write(linear_comment))
        self.play(Indicate(state_math[0][0][10]))
        self.wait(2)

        self.play(Indicate(state_math[1]))
        output_desc = Text("Final Hidden State for next step", font_size=18).next_to(
            state_math[1], DOWN
        )
        self.play(Write(output_desc))
        self.wait(3)
