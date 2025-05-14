import streamlit as st
import os
import re
import io
import time
import types
import base64
import logging
import multiprocessing
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from config import APP_VERSION, session_defaults
from file_processing import display_file, extract_chunks, process_chunk, estimate_total_chunks
from utils import authenticate, save_chat_history, initialize_app, save_config, logger, initialize_credentials, get_available_embedders, get_available_llms

# Try to import seaborn with fallback
try:
    import seaborn as sns
except ImportError as e:
    logger.error(f"Failed to import seaborn: {str(e)}. Plotting functionality will be disabled.")
    sns = None
    st.warning("Seaborn import failed. Plotting functionality is disabled. Please check your environment for compatible versions of numpy, scipy, and seaborn.")

def safe_message(content, is_user=False, key=None):
    try:
        if isinstance(content, types.GeneratorType):
            if st.session_state.debug_enabled:
                logger.warning("Generator object detected in message content, converting to string")
            try:
                content = "".join(content)
            except Exception as e:
                content = f"Error consuming generator: {str(e)}"
                logger.error(f"Failed to consume generator in message: {str(e)}")
        elif not isinstance(content, str):
            if st.session_state.debug_enabled:
                logger.warning(f"Non-string content type {type(content)} in message, converting to string")
            content = str(content)
        avatar = "🧟" if is_user else None
        st.chat_message("user" if is_user else "assistant", avatar=avatar).write(content)
    except Exception as e:
        logger.error(f"Error rendering message: {str(e)}")
        avatar = "🧟" if is_user else None
        st.chat_message("user" if is_user else "assistant", avatar=avatar).write(f"Error rendering message: {str(e)}")

def validate_latex(content):
    """Ensure LaTeX math delimiters are closed, passing through content unchanged otherwise."""
    if not isinstance(content, str):
        content = str(content)
    if st.session_state.debug_enabled:
        logger.debug(f"Validating LaTeX: '{content[:50]}...'")
    result = []
    i = 0
    in_inline_math = False
    in_display_math = False
    while i < len(content):
        if i + 1 < len(content) and content[i:i+2] == '$$':
            if in_display_math:
                in_display_math = False
            else:
                in_display_math = True
            result.append('$$')
            i += 2
        elif content[i] == '$' and not in_display_math:
            if in_inline_math:
                in_inline_math = False
            else:
                in_inline_math = True
            result.append('$')
            i += 1
        else:
            result.append(content[i])
            i += 1
    # Append closing delimiters if unclosed
    if in_inline_math:
        result.append('$')
        if st.session_state.debug_enabled:
            logger.debug("Appended closing $ to fix unclosed inline math")
    if in_display_math:
        result.append('$$')
        if st.session_state.debug_enabled:
            logger.debug("Appended closing $$ to fix unclosed display math")
    validated_content = ''.join(result)
    if st.session_state.debug_enabled and validated_content != content:
        logger.debug(f"LaTeX validated, changes made: '{validated_content[:50]}...'")
    return validated_content

def format_latex_content(content, append_to_messages=True):
    """Process content, splitting LaTeX into separate messages and returning non-LaTeX text."""
    if not isinstance(content, str):
        content = str(content)
    if st.session_state.debug_enabled:
        logger.debug(f"Formatting LaTeX content: '{content[:50]}...'")
    # Validate delimiters
    content = validate_latex(content)
    # Split into LaTeX and non-LaTeX segments
    segments = []
    i = 0
    current_text = ""
    while i < len(content):
        if i + 1 < len(content) and content[i:i+2] == '$$':
            # Display math
            end_idx = content.find('$$', i + 2)
            if end_idx == -1:
                current_text += content[i:]
                break
            if current_text:
                segments.append(('text', current_text))
                current_text = ""
            latex_content = content[i + 2:end_idx]
            segments.append(('latex', latex_content, True))
            i = end_idx + 2
        elif content[i] == '$':
            # Inline math
            end_idx = content.find('$', i + 1)
            if end_idx == -1:
                current_text += content[i:]
                break
            if current_text:
                segments.append(('text', current_text))
                current_text = ""
            latex_content = content[i + 1:end_idx]
            segments.append(('latex', latex_content, False))
            i = end_idx + 1
        else:
            current_text += content[i]
            i += 1
    if current_text:
        segments.append(('text', current_text))
    # Append segments to messages
    output_text = ""
    if append_to_messages:
        for seg_type, value, *args in segments:
            if seg_type == 'latex':
                display_mode = args[0]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": value,
                    "type": "latex",
                    "display_mode": display_mode
                })
                if st.session_state.debug_enabled:
                    logger.debug(f"Appended LaTeX message: '{value[:50]}...' (display_mode: {display_mode})")
            else:
                output_text += value
        if output_text and output_text.strip():
            st.session_state.messages.append({
                "role": "assistant",
                "content": output_text,
                "type": "text"
            })
            if st.session_state.debug_enabled:
                logger.debug(f"Appended text message: '{output_text[:50]}...'")
    else:
        output_text = "".join(value for seg_type, value, *args in segments if seg_type == 'text')
    if st.session_state.debug_enabled:
        logger.debug(f"Formatted output: '{output_text[:50]}...' (segments: {len(segments)})")
    return output_text

def render_gui():
    # Remove admin_credentials from session_defaults as it's handled in utils.py
    session_defaults_local = session_defaults.copy()
    session_defaults_local.pop('admin_credentials', None)
    for key, value in session_defaults_local.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Ensure credentials are initialized before login form
    initialize_credentials()

    st.set_page_config(
        page_title="RadioSport Chat",
        page_icon="🧟",
        layout="centered",
        initial_sidebar_state="collapsed",
        menu_items={
            'Report a Bug': "https://github.com/rkarikari/RadioSport-chat",
            'About': "Copyright © RNK, 2025 RadioSport. All rights reserved."
        }
    )
    st.markdown(
        """
        <style>
            input[type="checkbox"][id*="streaming_checkbox"] + div {
                background-color: #4A4A4A !important;
                border-radius: 4px;
                padding: 2px;
            }
            input[type="checkbox"][id*="streaming_checkbox"]:checked + div {
                background-color: #4A4A4A !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("RadioSport Chat")
    st.caption(f"Version {APP_VERSION}")

    with st.sidebar:
        if st.session_state.is_authenticated:
            st.title("🗂️ File Management")
            st.header("Upload Your Files")
            uploaded_files = st.file_uploader(
                "Select files",
                type=["pdf", "png", "jpg", "jpeg", "txt"],
                accept_multiple_files=True,
                key="file_uploader",
            )
            if uploaded_files:
                if any(f.type.startswith("image/") for f in uploaded_files):
                    st.session_state.last_uploaded_image = next(
                        (f for f in reversed(uploaded_files) if f.type.startswith("image/")), None
                    )
                if st.button("🚀 Add to Knowledge Base", type="primary"):
                    with st.status("Processing files...", expanded=True) as status_container:
                        if not hasattr(st.session_state, 'app'):
                            try:
                                logger.warning("st.session_state.app missing, reinitializing")
                                initialize_app()
                            except Exception as e:
                                status_container.error(f"Failed to reinitialize app: {str(e)}")
                                st.stop()
                        status_container.write("Extracting file content...")
                        chunks, extract_messages, extract_errors, temp_files = extract_chunks(uploaded_files, st.session_state.debug_enabled)
                        total_chunks = estimate_total_chunks(uploaded_files)
                        completed_chunks = multiprocessing.Value('i', 0)
                        status_container.write(f"Processing {total_chunks} chunks...")
                        file_stats = {}
                        all_messages = extract_messages[:]
                        all_errors = extract_errors[:]
                        status_messages = []
                        max_workers = min(multiprocessing.cpu_count() * 2, 16)
                        debug_mode = st.session_state.debug_enabled
                        progress_bar = status_container.progress(0.0, text="Processing chunks...")
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = [
                                executor.submit(
                                    process_chunk, st.session_state.app, file_name, chunk_text, data_type, debug_mode, completed_chunks, total_chunks
                                )
                                for file_name, chunk_text, data_type in chunks
                            ]
                            processed_chunks = 0
                            for future in concurrent.futures.as_completed(futures):
                                try:
                                    file_name, stat, status_message = future.result()
                                    if file_name not in file_stats:
                                        file_stats[file_name] = []
                                    file_stats[file_name].append(stat)
                                    status_messages.append(status_message)
                                    processed_chunks += 1
                                    progress_bar.progress(
                                        min(processed_chunks / total_chunks, 1.0),
                                        text=f"Processed {processed_chunks}/{total_chunks} chunks"
                                    )
                                except Exception as e:
                                    all_errors.append(f"Thread execution error: {str(e)}")
                                    logger.error(f"Thread execution error: {e}", exc_info=True)
                                    with completed_chunks.get_lock():
                                        completed_chunks.value += 1
                                    processed_chunks += 1
                                    status_messages.append(f"Processed chunk {completed_chunks.value}/{total_chunks}: Error ({str(e)})")
                                    progress_bar.progress(
                                        min(processed_chunks / total_chunks, 1.0),
                                        text=f"Processed {processed_chunks}/{total_chunks} chunks"
                                    )
                        for msg in status_messages:
                            status_container.write(msg)
                        for file_name, stats in file_stats.items():
                            total_chunks_file = len(stats)
                            successful_chunks = sum(1 for stat in stats if stat["success"])
                            skipped_chunks = sum(1 for stat in stats if stat["status"] == "Skipped (Duplicate)")
                            sum_duration = sum(stat["duration"] for stat in stats)
                            avg_duration = sum_duration / total_chunks_file if total_chunks_file > 0 else 0
                            embedding_dimension = next(
                                (stat["embedding_dimension"] for stat in stats if stat["success"]), 0
                            )
                            error_messages = [stat["error"] for stat in stats if not stat["success"]]
                            status = "Success" if successful_chunks + skipped_chunks == total_chunks_file else f"Failed: {', '.join(error_messages)}"
                            all_messages.append(
                                f"Processed {file_name}: {successful_chunks}/{total_chunks_file} chunks successful, "
                                f"{skipped_chunks} skipped (duplicates), "
                                f"Embedding Dimension: {embedding_dimension}, "
                                f"Total Duration: {sum_duration:.2f}s, "
                                f"Avg Duration: {avg_duration:.2f}s, "
                                f"Status: {status}"
                            )
                            if debug_mode:
                                logger.debug(f"File summary for {file_name}: {status}, {successful_chunks}/{total_chunks_file} chunks")
                        progress_bar.progress(1.0, text="Processing complete")
                        status_container.update(label="File processing complete", state="complete")
                        for msg in all_messages:
                            st.write(msg)
                        if all_errors:
                            st.subheader("Processing Errors")
                            for file_name in set(file_name for file_name, _, _ in chunks):
                                file_errors = [err for err in all_errors if file_name in err]
                                if file_errors:
                                    st.write(f"**{file_name}**:")
                                    for err in file_errors:
                                        st.error(err)
                        else:
                            st.success("✅ All files processed successfully")
            st.divider()
            st.subheader("📄 File Preview")
            if uploaded_files:
                file_names = [f.name for f in uploaded_files]
                selected_file = st.selectbox("Select file to preview", file_names)
                selected_file_obj = next(f for f in uploaded_files if f.name == selected_file)
                display_file(selected_file_obj)
            else:
                st.write("No files uploaded.")
            st.divider()

            with st.expander("📊 Plotting Graphs", expanded=False):
                st.write("Generate a plot based on input data or mathematical functions.")
                if sns is None:
                    st.error("Plotting is disabled due to missing or incompatible dependencies. Please ensure numpy, scipy, seaborn, and matplotlib are installed correctly.")
                else:
                    plot_type = st.selectbox(
                        "Select Plot Type",
                        ["Line Plot", "Bar Plot", "Scatter Plot", "Pie Chart", "Histogram", "Mathematical Function"],
                        key="plot_type_select"
                    )
                    if plot_type == "Mathematical Function":
                        function_mode = st.radio(
                            "Function Type",
                            ["Standard Function (y=f(x))", "Parametric Equations (x=f(t), y=f(t))"],
                            key="function_mode_select"
                        )
                        if function_mode == "Standard Function (y=f(x))":
                            function_input = st.text_input(
                                "Enter function (e.g., sin(x), x**2, cos(x))",
                                value="sin(x)",
                                key="function_input",
                                help="Use np.sin, np.cos, np.tan, np.exp, np.log, np.e for functions and constants."
                            )
                            x_range = st.text_input(
                                "X range (e.g., -10,10)",
                                value="-10,10",
                                key="x_range_input"
                            )
                            num_points = st.number_input(
                                "Number of points",
                                min_value=10,
                                max_value=1000,
                                value=100,
                                key="num_points_input"
                            )
                        else:  # Parametric Equations
                            parametric_functions = st.text_area(
                                "Enter parametric equations as x(t), y(t), label (one per line, e.g., cos(t), sin(t), Circle)",
                                value="cos(t), sin(t), Circle\nsin(2*t), sin(3*t), Lissajous",
                                key="parametric_functions_input",
                                help="Enter each curve as 'x(t), y(t), label'. Use np.sin, np.cos, np.tan, np.exp, np.log, np.e for functions and constants. Example: sin(t)*(np.exp(cos(t))-2*cos(4*t)-sin(t/12)**5), cos(t)*(np.exp(cos(t))-2*cos(4*t)-sin(t/12)**5), Butterfly"
                            )
                            t_range = st.text_input(
                                "T range (e.g., 0,100 for complex curves like Butterfly)",
                                value="0,100",
                                key="t_range_input"
                            )
                            num_points = st.number_input(
                                "Number of points",
                                min_value=10,
                                max_value=10000,
                                value=1000,
                                key="num_points_input_parametric"
                            )
                        data_input = None
                    else:
                        data_source = st.radio(
                            "Data Source",
                            ["Text Input", "CSV File"],
                            key="data_source_select"
                        )
                        if data_source == "Text Input":
                            data_input = st.text_area(
                                "Enter data (CSV format, e.g., label,value\\nA,10\\nB,20\\nC,25 for Pie Chart; value\\n10\\n20\\n25 for Histogram; x,y\\n1,10\\n2,20\\n3,25 for others):",
                                value="x,y\n1,10\n2,20\n3,25" if plot_type in ["Line Plot", "Bar Plot", "Scatter Plot"] else "label,value\nA,10\nB,20\nC,25" if plot_type == "Pie Chart" else "value\n10\n20\n25",
                                key="plot_data_input"
                            )
                            csv_file = None
                        else:
                            csv_file = st.file_uploader(
                                "Upload CSV file",
                                type=["csv"],
                                key="plot_csv_uploader"
                            )
                            data_input = None
                            if csv_file:
                                try:
                                    # Read CSV file into DataFrame
                                    df = pd.read_csv(csv_file)
                                    # Convert DataFrame to string for processing
                                    data_input = df.to_csv(index=False)
                                    st.write("**Uploaded CSV Preview:**")
                                    st.dataframe(df.head())
                                except Exception as e:
                                    st.error(f"Failed to read CSV file: {str(e)}")
                                    logger.error(f"CSV file read error: {str(e)}")
                                    data_input = None
                        if plot_type == "Histogram":
                            bins = st.number_input(
                                "Number of bins",
                                min_value=1,
                                max_value=100,
                                value=10,
                                key="bins_input"
                            )
                    plot_title = st.text_input("Plot Title", value="Sample Plot", key="plot_title_input")
                    x_label = st.text_input("X-Axis Label", value="X" if plot_type != "Pie Chart" else "", key="x_label_input")
                    y_label = st.text_input("Y-Axis Label", value="Y" if plot_type not in ["Pie Chart", "Histogram"] else "Frequency" if plot_type == "Histogram" else "", key="y_label_input")

                    if st.button("Generate Plot", key="generate_plot_btn"):
                        with st.spinner("Generating plot..."):
                            try:
                                plt.figure(figsize=(8, 6))
                                sns.set_style("whitegrid")
                                data_shape = None
                                if plot_type == "Mathematical Function":
                                    if function_mode == "Standard Function (y=f(x))":
                                        try:
                                            x_min, x_max = map(float, x_range.split(','))
                                        except ValueError:
                                            raise ValueError("Invalid x range format. Use 'min,max' (e.g., -10,10).")
                                        x = np.linspace(x_min, x_max, int(num_points))
                                        try:
                                            func = eval(f"lambda x: {function_input}", {"np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "e": np.e})
                                            y = func(x)
                                        except Exception as e:
                                            raise ValueError(f"Invalid function: {str(e)}. Ensure correct syntax (e.g., use np.sin, np.exp, np.e).")
                                        plt.plot(x, y, linewidth=2, label=function_input)
                                        data_shape = (len(x),)
                                    else:  # Parametric Equations
                                        try:
                                            t_min, t_max = map(float, t_range.split(','))
                                        except ValueError:
                                            raise ValueError("Invalid t range format. Use 'min,max' (e.g., 0,100).")
                                        t = np.linspace(t_min, t_max, int(num_points))
                                        colors = ['blue', 'red', 'green', 'purple', 'orange']  # Cycle through colors
                                        curves = []
                                        for i, line in enumerate(parametric_functions.strip().split('\n')):
                                            try:
                                                x_func, y_func, label = map(str.strip, line.split(','))
                                            except ValueError:
                                                raise ValueError(f"Invalid format for curve {i+1}. Use 'x(t), y(t), label'.")
                                            try:
                                                x_vals = eval(f"lambda t: {x_func}", {"np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "e": np.e, "t": t})(t)
                                                y_vals = eval(f"lambda t: {y_func}", {"np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "e": np.e, "t": t})(t)
                                            except SyntaxError as e:
                                                raise ValueError(f"Syntax error in parametric functions for curve {label}: {str(e)}. Check for correct use of np.sin, np.exp, np.e, and parentheses.")
                                            except Exception as e:
                                                raise ValueError(f"Invalid parametric functions for curve {label}: {str(e)}. Ensure correct syntax (e.g., use np.sin, np.exp, np.e, sin(t/12)**5).")
                                            curves.append((x_vals, y_vals, label, colors[i % len(colors)]))
                                        for x_vals, y_vals, label, color in curves:
                                            plt.plot(x_vals, y_vals, linewidth=2, label=label, color=color)
                                        data_shape = (len(t), len(curves))
                                else:
                                    if not data_input:
                                        raise ValueError("No data provided. Please enter data or upload a valid CSV file.")
                                    df = pd.read_csv(StringIO(data_input))
                                    if plot_type == "Pie Chart":
                                        if df.empty or len(df.columns) < 2:
                                            raise ValueError("Invalid data format. Please provide two columns (label, value).")
                                        labels, values = df.columns[0], df.columns[1]
                                        if not all(df[values].apply(lambda x: isinstance(x, (int, float)) and x >= 0)):
                                            raise ValueError("Values for Pie Chart must be non-negative numbers.")
                                        plt.pie(df[values], labels=df[labels], autopct='%1.1f%%', startangle=140)
                                        data_shape = df.shape
                                    elif plot_type == "Histogram":
                                        if df.empty or len(df.columns) < 1:
                                            raise ValueError("Invalid data format. Please provide at least one column (value).")
                                        values = df[df.columns[0]]
                                        if not all(values.apply(lambda x: isinstance(x, (int, float)))):
                                            raise ValueError("Values for Histogram must be numbers.")
                                        plt.hist(values, bins=int(bins), edgecolor='black')
                                        data_shape = (len(values),)
                                    else:
                                        if df.empty or len(df.columns) < 2:
                                            raise ValueError("Invalid data format. Please provide at least two columns (x, y).")
                                        x_col, y_col = df.columns[0], df.columns[1]
                                        x_data, y_data = df[x_col], df[y_col]
                                        if not (df[x_col].apply(lambda x: isinstance(x, (int, float))).all() and df[y_col].apply(lambda x: isinstance(x, (int, float))).all()):
                                            raise ValueError("X and Y values must be numbers.")
                                        if plot_type == "Line Plot":
                                            plt.plot(x_data, y_data, marker='o', linestyle='-', linewidth=2, markersize=8)
                                        elif plot_type == "Bar Plot":
                                            plt.bar(x_data, y_data, color='skyblue', edgecolor='black')
                                        elif plot_type == "Scatter Plot":
                                            plt.scatter(x_data, y_data, s=100, c='red', edgecolors='black')
                                        data_shape = df.shape
                                if plot_type != "Pie Chart":
                                    plt.title(plot_title, fontsize=14, pad=10)
                                    plt.xlabel(x_label, fontsize=12)
                                    plt.ylabel(y_label, fontsize=12)
                                    plt.grid(True)
                                    if plot_type == "Mathematical Function":
                                        plt.legend()  # Add legend for multiple curves
                                else:
                                    plt.title(plot_title, fontsize=14, pad=10)
                                plt.tight_layout()
                                buffer = BytesIO()
                                plt.savefig(buffer, format='png', dpi=100)
                                buffer.seek(0)
                                img_data = buffer.getvalue()
                                img_base64 = base64.b64encode(img_data).decode('utf-8')
                                img_str = f"data:image/png;base64,{img_base64}"
                                output = f"✅ Plot generated successfully: {plot_type} with title '{plot_title}'"
                                formatted_output = format_latex_content(output)
                                st.session_state.messages.append({"role": "assistant", "content": formatted_output, "type": "text"})
                                st.session_state.messages.append({"role": "assistant", "content": img_str, "type": "image"})
                                save_chat_history()
                                if st.session_state.debug_enabled:
                                    logger.debug(f"Plot generated: {plot_type}, data shape: {data_shape}")
                                    logger.debug(f"Appended message: {output[:50]}...")
                                    logger.debug(f"Appended plot image (base64 length: {len(img_str)})")
                                plt.close()
                                buffer.close()
                                st.rerun()
                            except Exception as e:
                                formatted_error = format_latex_content(f"❌ Plot generation failed: {str(e)}")
                                st.session_state.messages.append({"role": "assistant", "content": formatted_error, "type": "text"})
                                save_chat_history()
                                if st.session_state.debug_enabled:
                                    logger.debug(f"Plot generation failed: {str(e)}")
                                    logger.debug(f"Appended message: Plot generation failed: {str(e)[:50]}...")
                                st.error(f"Plot generation failed: {str(e)}")
                                if 'plt' in locals():
                                    plt.close()
                                st.rerun()

            if st.session_state.current_user == "admin":
                with st.expander("⚙️ Model Configuration", expanded=False):
                    st.write("Configure embedder and language model (Admin Only).")
                    available_embedders = get_available_embedders(st.session_state.debug_enabled)
                    selected_embedder = st.selectbox(
                        "Select Embedder",
                        available_embedders,
                        index=available_embedders.index(st.session_state.selected_embedder) if st.session_state.selected_embedder in available_embedders else 0,
                        key="embedder_select",
                        help="Select the embedding model for vectorization."
                    )
                    if selected_embedder != st.session_state.selected_embedder:
                        st.session_state.selected_embedder = selected_embedder
                        save_config()
                        logger.info(f"Embedder changed to {selected_embedder}, reinitializing app")
                        try:
                            initialize_app()
                            st.success(f"Embedder updated to {selected_embedder}. App reinitialized.")
                        except Exception as e:
                            st.error(f"Failed to reinitialize app with new embedder: {str(e)}")
                            logger.error(f"App reinitialization error: {str(e)}")
                        st.rerun()

                    available_llms = get_available_llms(st.session_state.debug_enabled)
                    selected_llm = st.selectbox(
                        "Select Language Model",
                        available_llms,
                        index=available_llms.index(st.session_state.selected_llm) if st.session_state.selected_llm in available_llms else 0,
                        key="llm_select",
                        help="Select the language model for chat responses."
                    )
                    if selected_llm != st.session_state.selected_llm:
                        st.session_state.selected_llm = selected_llm
                        save_config()
                        logger.info(f"Language model changed to {selected_llm}, reinitializing app")
                        try:
                            initialize_app()
                            st.success(f"Language model updated to {selected_llm}. App reinitialized.")
                        except Exception as e:
                            st.error(f"Failed to reinitialize app with new LLM: {str(e)}")
                            logger.error(f"App reinitialization error: {str(e)}")
                        st.rerun()

            st.divider()
            st.checkbox(
                "🔬 RAG Off",
                value=st.session_state.use_streaming,
                key="streaming_checkbox",
                on_change=lambda: (
                    st.session_state.update(use_streaming=st.session_state.streaming_checkbox),
                    save_config()
                )
            )

            if st.session_state.current_user == "admin":
                with st.expander("🧪 RAG Component Testing", expanded=False):
                    st.write("Test individual RAG components for troubleshooting.")
                    st.checkbox(
                        "🐞 Enable Debug Mode",
                        value=st.session_state.debug_enabled,
                        key="debug_checkbox",
                        on_change=lambda: st.session_state.update(debug_enabled=st.session_state.debug_checkbox)
                    )
                    test_text = st.text_area("Enter text to test embedding:", value="The whole Duty of man. Fear God, keep his commands.")
                    if st.button("Test Embedding", key="test_embed_btn"):
                        with st.spinner("Testing embedding..."):
                            try:
                                if not hasattr(st.session_state, 'app'):
                                    st.session_state.messages.append({"role": "assistant", "content": "st.session_state.app is not initialized. Please restart the app.", "type": "text"})
                                    save_chat_history()
                                    if st.session_state.debug_enabled:
                                        logger.debug("Appended message: st.session_state.app is not initialized")
                                    st.rerun()
                                else:
                                    result = st.session_state.app.test_embedding(test_text)
                                    if result["success"]:
                                        output = (
                                            f"✅ Embedding successful - Dimension: {result['embedding_dimension']}\n"
                                            f"Sample of embedding vector: {result['embedding_sample']}\n"
                                            f"Process took {result['duration']:.4f} seconds"
                                        )
                                        format_latex_content(output)
                                        save_chat_history()
                                        if st.session_state.debug_enabled:
                                            logger.debug(f"Appended message: {output[:50]}...")
                                    else:
                                        formatted_error = format_latex_content(f"❌ Embedding failed: {result['error']}")
                                        st.session_state.messages.append({"role": "assistant", "content": formatted_error, "type": "text"})
                                        save_chat_history()
                                        if st.session_state.debug_enabled:
                                            logger.debug(f"Appended message: Embedding failed: {result['error'][:50]}...")
                                    st.rerun()
                            except Exception as e:
                                formatted_error = format_latex_content(f"Test failed with error: {str(e)}")
                                st.session_state.messages.append({"role": "assistant", "content": formatted_error, "type": "text"})
                                save_chat_history()
                                if st.session_state.debug_enabled:
                                    logger.debug(f"Appended message: Test failed: {str(e)[:50]}...")
                                st.rerun()
                    if st.button("Test Minimal Streaming", key="test_minimal_stream_btn"):
                        with st.spinner("Testing minimal streaming..."):
                            try:
                                def stream_data():
                                    for word in ["Hello", "world", "this", "is", "a", "test", "with", "math", "$\\sum_{i=1}^{n} i$"]:
                                        yield word + " "
                                        time.sleep(0.5)
                                st.session_state.messages.append({"role": "assistant", "content": "Testing minimal streaming...", "type": "text"})
                                save_chat_history()
                                if st.session_state.debug_enabled:
                                    logger.debug("Appended message: Testing minimal streaming...")
                                text = ""
                                if st.session_state.use_streaming:
                                    progress_bar = st.progress(0)
                                    chunk_count = 0
                                    total_chunks_estimated = 9
                                    st.session_state.streaming_session_id += 1
                                    session_id = st.session_state.streaming_session_id
                                    placeholder = st.empty()
                                    try:
                                        for chunk in st.write_stream(stream_data()):
                                            chunk_count += 1
                                            text += chunk
                                            formatted_text = format_latex_content(text, append_to_messages=False)
                                            placeholder.markdown(f"**Assistant:** {formatted_text}", unsafe_allow_html=True)
                                            if st.session_state.debug_enabled:
                                                logger.debug(
                                                    f"Streamed chunk {chunk_count} at {time.time():.2f}s: "
                                                    f"'{chunk[:50]}...' (len={len(chunk)}, total_len={len(text)}, session_id={session_id})"
                                                )
                                            progress_bar.progress(min(chunk_count / total_chunks_estimated, 1.0))
                                        format_latex_content(text)
                                        st.session_state.messages.append({"role": "assistant", "content": "Minimal streaming test completed.", "type": "text"})
                                        save_chat_history()
                                        if st.session_state.debug_enabled:
                                            logger.debug(f"Appended message: {text[:50]}...")
                                            logger.debug("Appended message: Minimal streaming test completed")
                                        progress_bar.progress(1.0)
                                        st.rerun()
                                    except Exception as e:
                                        logger.error(f"Minimal streaming error: {str(e)}")
                                        formatted_error = format_latex_content(f"Minimal streaming error: {str(e)}")
                                        placeholder.markdown(f"**Assistant:** {formatted_error}", unsafe_allow_html=True)
                                        st.session_state.messages.append({"role": "assistant", "content": formatted_error, "type": "text"})
                                        st.session_state.messages.append({"role": "assistant", "content": "Minimal streaming test failed.", "type": "text"})
                                        save_chat_history()
                                        if st.session_state.debug_enabled:
                                            logger.debug(f"Appended message: Minimal streaming error: {str(e)[:50]}...")
                                            logger.debug("Appended message: Minimal streaming test failed")
                                        progress_bar.progress(1.0)
                                        st.rerun()
                                else:
                                    start_time = time.time()
                                    for chunk in stream_data():
                                        text += chunk
                                    format_latex_content(text)
                                    st.session_state.messages.append({"role": "assistant", "content": "Minimal streaming test completed.", "type": "text"})
                                    save_chat_history()
                                    if st.session_state.debug_enabled:
                                        logger.debug(
                                            f"Rendered full response (non-streaming) at {time.time() - start_time:.2f}s: "
                                            f"'{text[:50]}...' (len={len(text)})"
                                        )
                                        logger.debug(f"Appended message: {text[:50]}...")
                                        logger.debug("Appended message: Minimal streaming test completed")
                                    st.rerun()
                            except Exception as e:
                                formatted_error = format_latex_content(f"Minimal streaming test failed: {str(e)}")
                                st.session_state.messages.append({"role": "assistant", "content": formatted_error, "type": "text"})
                                save_chat_history()
                                if st.session_state.debug_enabled:
                                    logger.debug(f"Appended message: Minimal streaming test failed: {str(e)[:50]}...")
                                st.rerun()
                    st.divider()
                    with st.container():
                        col1, col2 = st.columns([2, 1], gap="small")
                        with col1:
                            st.button(
                                "🔄 Reset Session State",
                                disabled=not st.session_state.confirm_reset_session,
                                on_click=(lambda: (
                                    st.session_state.update({k: v for k, v in session_defaults.items()}),
                                    os.remove("chat_history.json") if os.path.exists("chat_history.json") else None,
                                    os.remove("config.json") if os.path.exists("config.json") else None,
                                    logger.info("Session state reset"),
                                    st.rerun()
                                ))
                            )
                        with col2:
                            st.checkbox(
                                "Confirm Reset",
                                value=False,
                                key="confirm_reset_session"
                            )

            st.divider()
            if st.button("🔒 Logout"):
                st.session_state.is_authenticated = False
                st.session_state.debug_enabled = False
                st.session_state.use_streaming = False
                st.session_state.last_uploaded_image = None
                st.session_state.show_login_panel = True
                st.session_state.debug_sessions = []
                st.session_state.current_user = None
                save_config()
                logger.info("User logged out")
                st.rerun()
        elif st.session_state.show_login_panel:
            st.subheader("🔐 Admin Login")
            with st.form(key="login_form"):
                username = st.text_input("Username", help="Enter your username")
                password = st.text_input("Password", type="password", help="Enter your password")
                submit_button = st.form_submit_button(label="Login")
                if submit_button:
                    if not username or not password:
                        st.error("Username and password cannot be empty")
                    elif username not in ["admin", "user"]:
                        st.error("Invalid username")
                        logger.warning(f"Invalid username entered: {username}")
                    else:
                        if authenticate(username, password):
                            st.session_state.is_authenticated = True
                            st.session_state.debug_enabled = False
                            st.session_state.show_login_panel = False
                            st.session_state.current_user = username
                            st.success("Login successful!")
                            logger.info(f"Login successful for {username}")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")

    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            safe_message(msg["content"], is_user=True, key=str(i))
        else:
            content = msg["content"]
            msg_type = msg.get("type", "text")
            if isinstance(content, types.GeneratorType):
                if st.session_state.debug_enabled:
                    logger.warning("Generator object detected in assistant message content, converting to string")
                try:
                    content = "".join(content)
                except Exception as e:
                    content = f"Error consuming generator: {str(e)}"
                    logger.error(f"Failed to consume generator in assistant message: {str(e)}")
            elif not isinstance(content, str):
                if st.session_state.debug_enabled:
                    logger.warning(f"Non-string content type {type(content)} in assistant message, converting to string")
                content = str(content)
            try:
                if msg_type == "image":
                    st.image(content, use_container_width=True)
                elif msg_type == "latex":
                    display_mode = msg.get("display_mode", False)
                    st.latex(content)
                    if st.session_state.debug_enabled:
                        logger.debug(f"Rendered LaTeX: '{content[:50]}...' (display_mode: {display_mode})")
                else:
                    st.markdown(f"**Assistant:** {content}", unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Failed to render assistant message: {str(e)}")
                st.markdown(f"**Assistant:** {content} (Rendering failed: {str(e)})", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.checkbox(
            "Confirm: Clear chat history",
            value=False,
            key="confirm_clear_chat"
        )
        if st.button("🧹 Clear Chat History", disabled=not st.session_state.confirm_clear_chat):
            st.session_state.messages = []
            st.session_state.debug_sessions = []
            st.session_state.streaming_session_id = 0
            st.session_state.streaming_active = False
            if os.path.exists("chat_history.json"):
                os.remove("chat_history.json")
            logger.info("Chat history cleared")
            st.rerun()

    if st.session_state.debug_enabled and st.session_state.is_authenticated and st.session_state.current_user == "admin":
        st.divider()
        st.subheader("🔍 RAG Pipeline Debug Information")
        try:
            if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, 'get_debug_info'):
                debug_info = st.session_state.app.get_debug_info()
            else:
                st.error("Debug information unavailable: Application not initialized.")
                logger.error("Failed to access debug info: st.session_state.app or get_debug_info not available")
                debug_info = {"add_operations": [], "query_operations": []}
        except Exception as e:
            st.error(f"Error accessing debug information: {str(e)}")
            logger.error(f"Error accessing debug info: {str(e)}")
            debug_info = {"add_operations": [], "query_operations": []}
        debug_tab1, debug_tab2, debug_tab3 = st.tabs(
            ["Current Query", "Add Operations", "DB Stats"]
        )
        with debug_tab1:
            st.write("**Recent Queries:**")
            if st.session_state.debug_sessions:
                for i, session in enumerate(reversed(st.session_state.debug_sessions[-5:])):
                    st.write(f"**Query {len(st.session_state.debug_sessions) - i}:**")
                    st.write(f"**Prompt:** {session.get('prompt', 'No prompt available')}")
                    st.write(f"**Response:** {session.get('response', 'No response available')}")
                    st.write("**Streaming Info:**")
                    streaming_info = session.get('streaming_info', {})
                    st.write(f"- Response Type: {streaming_info.get('response_type', 'Unknown')}")
                    st.write(f"- Total Chunks: {streaming_info.get('chunk_count', 0)}")
                    if streaming_info.get('chunks_received'):
                        st.write("- Chunks Received:")
                        for chunk_info in streaming_info['chunks_received'][:5]:
                            st.write(
                                f"  - Chunk {chunk_info['chunk_number']}: '{chunk_info['content']}' "
                                f"(len={chunk_info['length']}, time={chunk_info['timestamp']:.2f}s)"
                            )
                    st.write("**Retrieved Documents:** Not available due to Embedchain API limitations.")
                    st.write("---")
            else:
                st.write("No queries recorded yet.")
        with debug_tab2:
            st.write("**File Embedding Reports:**")
            if debug_info.get("add_operations", []):
                for op in debug_info["add_operations"]:
                    st.write(
                        f"- [{op['timestamp']}] File: {op.get('file_name', 'Unknown')} (Type: {op['data_type']}), "
                        f"Status: {op.get('status', 'Unknown')}, "
                        f"Chunks: {op.get('successful_chunks', 1 if op.get('success', False) else 0)}/{op.get('total_chunks', 1)}, "
                        f"Embedding Dimension: {op['embedding_dimension']}, "
                        f"Total Duration: {op.get('total_duration', op.get('duration', 0)):.2f}s, "
                        f"Avg Duration: {op.get('avg_duration', op.get('duration', 0)):.2f}s, "
                        f"Snippet: {op['text_snippet']}"
                    )
            else:
                st.write("No file embedding reports recorded.")
        with debug_tab3:
            st.write("**Vector DB Stats:**")
            try:
                if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, "db"):
                    if st.session_state.debug_enabled:
                        logger.debug("Accessing DB stats")
                    count = st.session_state.app.db.count()
                    st.write(f"📈 Documents in DB: {count}")
                    if st.session_state.debug_enabled:
                        logger.debug(f"Current DB document count: {count}")
            except Exception as e:
                st.error(f"Error accessing vector DB: {str(e)}")
                logger.error(f"Vector DB access error: {e}")