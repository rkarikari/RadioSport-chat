import streamlit as st

st.markdown(
    """
    <script src="static/mathjax/tex-chtml.js" id="MathJax-script"></script>
    <script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            chtml: {
                scale: 1
            },
            startup: {
                ready: function () {
                    console.log("MathJax is ready");
                    MathJax.startup.defaultReady();
                }
            }
        };
    </script>
    """,
    unsafe_allow_html=True
)

st.write("Test inputs:")
st.markdown(r"This is a test with $x^2 + y^2 = z^2$", unsafe_allow_html=True)
st.markdown(r"Here is an equation: $$E = mc^2$$", unsafe_allow_html=True)
st.markdown(r"Solve for $x$: \[ x^2 - 4 = 0 \]", unsafe_allow_html=True)
st.markdown(r"Response: $$c^2 = a^2 + b^2 $$", unsafe_allow_html=True)