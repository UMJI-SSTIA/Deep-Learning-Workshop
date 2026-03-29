from manim import *
import numpy as np


class Convolution2D(Scene):
    def construct(self):
        # --- Titles and Formulas ---
        title = Tex("2D Convolution Visualization").to_edge(UP)
        formula = (
            MathTex(r"H(x, y) = \sum_{i} \sum_{j} F(x+i, y+j) \cdot G(i, j)")
            .next_to(title, DOWN, buff=0.3)
            .scale(0.8)
        )
        self.add(title, formula)

        # --- Define Data ---
        image_data = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 255, 100, 50, 0],
                [0, 100, 200, 100, 0],
                [0, 50, 100, 50, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        kernel_data = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 10.0
        output_shape = (3, 3)

        # --- Create Matrices ---
        image_m = Matrix(image_data).scale(0.6).to_edge(LEFT, buff=1)
        image_label = Tex("Image $F$").next_to(image_m, UP)

        kernel_m = Matrix(kernel_data).scale(0.6).next_to(image_m, RIGHT, buff=1)
        kernel_label = Tex("Kernel $G$").next_to(kernel_m, UP)

        output_m = Matrix(np.zeros(output_shape)).scale(0.6).to_edge(RIGHT, buff=1)
        output_label = Tex("Output $H$").next_to(output_m, UP)

        self.add(
            image_m,
            image_label,
            kernel_m,
            kernel_label,
            output_m.brackets,
            output_label,
        )

        # Helper to get a cell from a Matrix object
        def get_matrix_cell(matrix, row, col):
            # matrix.get_rows() returns a list of VGroups (rows)
            return matrix.get_rows()[row - 1][col - 1]

        # --- Sliding Window Setup ---
        # We want to highlight the 3x3 area. We use the top-left and bottom-right cells.
        ul_cell = get_matrix_cell(image_m, 1, 1)
        dr_cell = get_matrix_cell(image_m, 3, 3)

        sliding_window = SurroundingRectangle(
            VGroup(get_matrix_cell(image_m, 1, 1), get_matrix_cell(image_m, 3, 3)),
            color=ORANGE,
            buff=0.1,
        )

        arrow = Arrow(kernel_m.get_right(), output_m.get_left(), color=YELLOW)
        self.play(Create(sliding_window), GrowArrow(arrow))

        # --- Animation Loop ---
        for r in range(output_shape[0]):
            for c in range(output_shape[1]):
                # Move window to cover the current 3x3 block
                # The block starts at row r+1, col c+1
                current_block = VGroup(
                    get_matrix_cell(image_m, r + 1, c + 1),
                    get_matrix_cell(image_m, r + 3, c + 3),
                )

                # Calculate value (simplified for visual)
                window_data = image_data[r : r + 3, c : c + 3]
                val = np.sum(np.multiply(window_data, kernel_data))

                self.play(
                    sliding_window.animate.move_to(current_block.get_center()),
                    run_time=0.5,
                )

                # Target cell in output matrix
                target_cell = get_matrix_cell(output_m, r + 1, c + 1)
                res_val = (
                    DecimalNumber(val, num_decimal_places=1)
                    .scale(0.5)
                    .move_to(target_cell)
                )

                highlighter = SurroundingRectangle(target_cell, color=RED)

                self.play(Create(highlighter), FadeIn(res_val), run_time=0.4)
                self.play(FadeOut(highlighter), run_time=0.2)

        self.wait(2)


class WhiskerMaxPooling(Scene):
    def construct(self):
        # --- Titles and Setup ---
        title = Text("Max Pooling: Matrix Spatial Reduction", font_size=40).to_edge(UP)
        self.play(Write(title))

        # Define the Input Feature Map (4x4 Matrix)
        # 0.9 represents the strongest whisker activation
        input_data = [
            [0.1, 0.2, 0.1, 0.0],
            [0.1, 0.9, 0.3, 0.1],
            [0.0, 0.2, 0.1, 0.0],
            [0.1, 0.0, 0.0, 0.1],
        ]

        # Create Input Matrix
        input_matrix = Matrix(input_data).scale(0.7).shift(LEFT * 3)
        matrix_label = Text("4x4 Feature Map (Whiskers)", font_size=24).next_to(
            input_matrix, UP
        )

        self.play(FadeIn(input_matrix), FadeIn(matrix_label))
        self.wait(1)

        # Highlight the key feature (the '0.9' whisker detection)
        # Matrix.get_entries() returns a flat list. Index 5 is Row 2, Col 2.
        strong_feature = input_matrix.get_entries()[5]
        self.play(strong_feature.animate.set_color(RED), run_time=1)

        # --- Prepare Output Matrix ---
        # Initialize a 2x2 Matrix with placeholders
        output_matrix = Matrix([[0.0, 0.0], [0.0, 0.0]]).scale(0.7).shift(RIGHT * 3)
        output_label = Text("2x2 Pooled Output", font_size=24).next_to(
            output_matrix, UP
        )

        # Calculate Window Geometry
        # We find the distance between matrix entries to size the sliding window
        entries = input_matrix.get_entries()
        dx = entries[1].get_x() - entries[0].get_x()
        dy = entries[0].get_y() - entries[4].get_y()

        # Pooling Window (Yellow Rectangle)
        pool_window = Rectangle(
            width=dx * 1.9, height=dy * 1.9, color=YELLOW, stroke_width=4
        )
        # Center the window on the top-left 2x2 quadrant
        first_quad_center = VGroup(
            entries[0], entries[1], entries[4], entries[5]
        ).get_center()
        pool_window.move_to(first_quad_center)

        self.play(
            FadeIn(output_matrix.brackets), FadeIn(output_label), Create(pool_window)
        )
        self.wait(1)

        # --- Max Pooling Step-by-Step Logic ---
        # quadrants maps input indices to output indices and defines window movement
        quadrants = [
            {"input_idx": [0, 1, 4, 5], "out_idx": 0, "shift_dir": ORIGIN},
            {"input_idx": [2, 3, 6, 7], "out_idx": 1, "shift_dir": RIGHT * dx * 2},
            {
                "input_idx": [8, 9, 12, 13],
                "out_idx": 2,
                "shift_dir": DOWN * dy * 2 + LEFT * dx * 2,
            },
            {"input_idx": [10, 11, 14, 15], "out_idx": 3, "shift_dir": RIGHT * dx * 2},
        ]

        for i, quad in enumerate(quadrants):
            # 1. Slide the window
            if i > 0:
                self.play(pool_window.animate.shift(quad["shift_dir"]), run_time=0.6)

            # 2. Identify the active quadrant
            active_entries = VGroup(*[entries[idx] for idx in quad["input_idx"]])
            self.play(Indicate(active_entries, color=YELLOW, scale_factor=1.1))

            # 3. Extract the maximum value from the data
            current_vals = [input_data[idx // 4][idx % 4] for idx in quad["input_idx"]]
            max_val = max(current_vals)

            # 4. Animate the Max value flying to the output matrix
            target_entry = output_matrix.get_entries()[quad["out_idx"]]
            moving_val = DecimalNumber(max_val, num_decimal_places=1).scale(0.7)
            moving_val.move_to(active_entries.get_center())

            self.play(
                moving_val.animate.move_to(target_entry.get_center()), run_time=0.8
            )
            # Ensure the output entry is highlighted
            self.add(moving_val.set_color(YELLOW))
            self.wait(0.5)

        # --- Conclusion / Summary ---
        self.play(FadeOut(pool_window))

        summary = (
            VGroup(
                Text("Max Pooling Principles:", font_size=25, color=BLUE),
                Text("1. Downsampling: 4x4 matrix shrinks to 2x2.", font_size=20),
                Text("2. Invariance: The large signal survived.", font_size=20),
                Text(
                    "The model stays robust to minor feature shifts.",
                    font_size=20,
                    color=GRAY,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(DOWN)
        )

        self.play(Write(summary))

        self.wait(3)


class CNNArchitecture(Scene):
    def construct(self):
        # Configuration for data blocks
        data_config = {
            "stroke_color": WHITE,
            "stroke_width": 1,
            "fill_opacity": 0.6,
        }

        # --- STEP 1: Input Image ---
        self.next_section("Step 1: Input")
        title = Text("CNN Architecture: The Grand Tour", font_size=36).to_edge(UP)
        self.play(Write(title))

        input_box = Rectangle(width=3, height=3, color=BLUE, **data_config)
        input_box.set_fill(BLUE_E)
        # Add grid lines to represent pixel structure
        grid = NumberPlane(
            x_range=[0, 8, 1],
            y_range=[0, 8, 1],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 0.5,
                "stroke_opacity": 1,
            },
        ).replace(input_box)
        input_grp = VGroup(input_box, grid).shift(LEFT * 5)

        # Explicit labels defined in prompt
        input_label = MathTex(r"32 \times 32 \times 3", font_size=24).next_to(
            input_grp, DOWN
        )
        feat_low_label = Text(
            "Low-Level Features\n(Edges, fur patches)", font_size=20, color=BLUE_C
        ).next_to(input_grp, UP)

        self.play(FadeIn(input_grp), Write(input_label))
        self.play(Write(feat_low_label))
        self.wait(1)

        # --- STEP 2: Convolutional Layers & High-Level Features ---
        self.next_section("Step 2: Features")

        # Define a VGroup for feature maps
        feature_maps = VGroup()
        map_colors = [TEAL_E, GREEN_E, YELLOW_E]
        shapes = [(2.5, 16), (2.0, 32), (1.5, 64)]  # (size, depth)

        for i, (size, depth) in enumerate(shapes):
            map_stack = VGroup()
            for j in range(3):  # visual stack depth
                rect = Rectangle(
                    width=size, height=size, color=map_colors[i], **data_config
                )
                rect.shift(RIGHT * (j * 0.1) + UP * (j * 0.1))
                map_stack.add(rect)
            map_stack.move_to(LEFT * 2.5 + RIGHT * (i * 2))
            feature_maps.add(map_stack)

        feat_high_label = Text(
            "High-Level Features",
            font_size=20,
            color=YELLOW_C,
        ).next_to(feature_maps, UP)

        # Simple arrows showing data flow
        arrow1 = Arrow(input_grp.get_right(), feature_maps[0].get_left(), buff=0.2)
        arrow2 = Arrow(
            feature_maps[0].get_right(), feature_maps[1].get_left(), buff=0.2
        )
        arrow3 = Arrow(
            feature_maps[1].get_right(), feature_maps[2].get_left(), buff=0.2
        )
        flow_arrows = VGroup(arrow1, arrow2, arrow3)

        self.play(
            FadeOut(feat_low_label),
            Write(feat_high_label),
            Create(flow_arrows),
            FadeIn(feature_maps, shift=RIGHT),
        )
        self.wait(2)

        # --- STEP 3: The Flattening (Transition) ---
        self.next_section("Step 3: Flattening")

        # Create the small, deep volume explicitly requested (e.g., 4x4x64)
        deep_vol_box = Rectangle(width=1.0, height=1.0, color=PURPLE, **data_config)
        deep_vol_box.set_fill(PURPLE_E)

        # Visualizing depth by stacking many layers slightly offset
        deep_vol_stack = VGroup()
        for i in range(8):
            d = Rectangle(width=1.0, height=1.0, color=PURPLE, **data_config)
            d.set_fill(PURPLE_E)
            d.shift(RIGHT * (i * 0.05) + UP * (i * 0.05))
            deep_vol_stack.add(d)

        deep_vol_grp = deep_vol_stack.move_to(feature_maps[2].get_center())
        vol_label = MathTex(r"4 \times 4 \times 64", font_size=22).next_to(
            deep_vol_grp, DOWN
        )

        # The Flattened Vector (concept vector)
        vector_dots = VGroup(
            *[Circle(radius=0.08, color=WHITE, fill_opacity=1) for _ in range(12)]
        )
        vector_dots.arrange(DOWN, buff=0.15).shift(RIGHT * 4.5)
        vec_label = MathTex(r"1024 \text{ Vector}", font_size=22).next_to(
            vector_dots, DOWN
        )
        vec_concept_label = Text(
            "Concepts:\n[fur, 4 legs, ears...]", font_size=18, color=GREY_C
        ).next_to(vector_dots, UP)

        self.play(
            FadeOut(feat_high_label),
            FadeOut(flow_arrows),
            FadeOut(feature_maps[:2]),  # Keep the last feature map to transform it
            ReplacementTransform(feature_maps[2], deep_vol_grp),
            Write(vol_label),
        )
        self.wait(1)

        self.play(
            ReplacementTransform(deep_vol_grp, vector_dots),
            Transform(vol_label, vec_label),
            Write(vec_concept_label),
        )
        self.wait(2)

        # --- STEP 4: FC Layer and The Classifier ---
        self.next_section("Step 4: Classifier")

        output_labels = (
            VGroup(
                Text("98% Cat", color=GREEN, font_size=26),
                Text("2% Dog", color=RED, font_size=26),
            )
            .arrange(DOWN, buff=0.4)
            .shift(RIGHT * 5.5)
        )

        # Draw dense connections (FC)
        fc_connections = VGroup()
        for dot in vector_dots:
            fc_connections.add(
                Line(
                    dot.get_right(),
                    output_labels[0].get_left(),
                    stroke_width=0.5,
                    stroke_opacity=0.2,
                    color=GRAY,
                )
            )
            fc_connections.add(
                Line(
                    dot.get_right(),
                    output_labels[1].get_left(),
                    stroke_width=0.5,
                    stroke_opacity=0.2,
                    color=GRAY,
                )
            )

        self.play(
            FadeOut(vec_concept_label),
            FadeOut(vec_label),
            FadeOut(vol_label),
            Create(fc_connections),
            Write(output_labels),
        )
        self.wait(2)

        # --- STEP 5: Final Full Architecture View ---
        self.next_section("Final Architecture")

        # We need to shrink everything and arrange it horizontally for the final view
        final_group = VGroup(
            input_grp,
            input_label,
            feature_maps[0],
            feature_maps[1],
            feature_maps[2],
            vector_dots,
            vol_label,  # vol_label was transformed to vec_label
            fc_connections,
            output_labels,
        )

        final_title = Text(
            "Full CNN Architecture: Image to Concept", font_size=32
        ).to_edge(UP)

        self.play(
            FadeOut(title),
            FadeOut(vector_dots),  # To replace with a cleaner final representation
            FadeOut(fc_connections),
            FadeOut(input_grp),
            FadeOut(input_label),
            FadeOut(output_labels),
            FadeOut(output_labels),
        )
        self.wait(0.5)

        # Re-creating static simplified components for the full view
        s = 0.6  # Scale factor
        f_input = input_grp.copy().scale(s).move_to(LEFT * 5.5)
        f_input_l = MathTex(r"32{\times}32{\times}3", font_size=18).next_to(
            f_input, DOWN, buff=0.1
        )

        # Re-create feature stacks at final positions
        f_maps = VGroup()
        x_pos = [-3.5, -1.8, -0.2]
        for i, (size, depth) in enumerate(shapes):
            map_stack = VGroup()
            for j in range(3):
                rect = Rectangle(
                    width=size * s, height=size * s, color=map_colors[i], **data_config
                )
                rect.shift(RIGHT * (j * 0.05) + UP * (j * 0.05))
                map_stack.add(rect)
            map_stack.move_to(RIGHT * x_pos[i])
            f_maps.add(map_stack)

        # Labels for feature maps
        f_map_ls = VGroup(
            MathTex(r"28{\times}28{\times}16", font_size=16).next_to(
                f_maps[0], DOWN, buff=0.1
            ),
            MathTex(r"14{\times}14{\times}32", font_size=16).next_to(
                f_maps[1], DOWN, buff=0.1
            ),
            MathTex(r"4{\times}4{\times}64", font_size=16).next_to(
                f_maps[2], DOWN, buff=0.1
            ),
        )

        f_vector = VGroup(
            *[Circle(radius=0.04, color=WHITE, fill_opacity=1) for _ in range(20)]
        )
        f_vector.arrange(DOWN, buff=0.08).shift(RIGHT * 2.0)
        f_vec_l = MathTex(r"1024", font_size=18).next_to(f_vector, DOWN, buff=0.1)

        f_output = (
            VGroup(
                Text("Cat", color=GREEN, font_size=20),
                Text("Dog", color=RED, font_size=20),
            )
            .arrange(DOWN, buff=0.5)
            .shift(RIGHT * 4.5)
        )
        f_output_l = Text("Output", font_size=18, color=GREY).next_to(
            f_output, DOWN, buff=0.2
        )

        # Connecting Arrows (Static)
        arrows = VGroup(
            Arrow(
                f_input.get_right(),
                f_maps[0].get_left(),
                buff=0.1,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            ),
            Arrow(
                f_maps[0].get_right(),
                f_maps[1].get_left(),
                buff=0.1,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            ),
            Arrow(
                f_maps[1].get_right(),
                f_maps[2].get_left(),
                buff=0.1,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            ),
            Arrow(
                f_maps[2].get_right(),
                f_vector.get_left(),
                buff=0.1,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            ),
            Arrow(
                f_vector.get_right(),
                f_output.get_left(),
                buff=0.1,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            ),
        )

        # Section Labels
        section_labels = VGroup(
            Text("INPUT", font_size=14, color=BLUE_B).next_to(f_input, UP, buff=0.4),
            Text("FEATURE EXTRACTION (CONV+POOL)", font_size=14, color=TEAL_B).next_to(
                f_maps[1], UP, buff=0.4
            ),
            Text("FLATTEN", font_size=14, color=PURPLE_B).next_to(
                f_vec_l, UP, buff=3.6
            ),
            Text("CLASSIFIER", font_size=14, color=YELLOW_B).next_to(
                f_output, UP, buff=0.4
            ),
        )

        self.play(
            FadeIn(final_title),
            FadeIn(f_input),
            FadeIn(f_input_l),
            FadeIn(f_maps),
            FadeIn(f_map_ls),
            FadeIn(f_vector),
            FadeIn(f_vec_l),
            FadeIn(f_output),
            FadeIn(f_output_l),
            Create(arrows),
            Write(section_labels),
        )
        self.wait(5)
