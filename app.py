import streamlit as st
from PIL import Image
import torch

from predict import load_model, tf, BIN_MAP
from logger import log_scan

ASK_BELOW = 0.80  # isse kam confidence par agent sawaal poochta hai

# Confusing pairs ke liye follow-up sawaal: {option text: final label}
QUESTIONS = {
    frozenset(["glass", "plastic"]): (
        "Ye cheez kaisi mehsoos hoti hai?",
        {"Bhaari aur thandi, toote to nukile tukde": "glass",
         "Halki aur pichakne wali (dabane par dab jaye)": "plastic"},
    ),
    frozenset(["paper", "cardboard"]): (
        "Kya isme moti, lehardar (corrugated) layer hai?",
        {"Haan, moti aur lehardar": "cardboard",
         "Nahi, patla kagaz": "paper"},
    ),
    frozenset(["paper", "trash"]): (
        "Kya ye saaf aur sukha hai?",
        {"Haan, saaf aur sukha": "paper",
         "Nahi, gila ya tel/khane se gandha": "trash"},
    ),
    frozenset(["plastic", "trash"]): (
        "Ye kis type ka plastic hai?",
        {"Kadak container/bottle (recycle number ke saath)": "plastic",
         "Naram wrapper ya chips/biscuit packet": "trash"},
    ),
    frozenset(["plastic", "metal"]): (
        "Kya ye chamakdaar aur thanda dhatu jaisa hai?",
        {"Haan, dhatu jaisa (magnet ya can)": "metal",
         "Nahi, plastic jaisa": "plastic"},
    ),
    frozenset(["cardboard", "trash"]): (
        "Kya ye saaf aur sukha hai?",
        {"Haan, saaf aur sukha": "cardboard",
         "Nahi, gila ya gandha": "trash"},
    ),
}


def agent_decide(results):
    """Confidence kam ho to follow-up sawaal (question, options) lautata hai."""
    (l1, p1), (l2, _) = results[0], results[1]
    if p1 >= ASK_BELOW:
        return None
    pair = frozenset([l1, l2])
    if pair in QUESTIONS:
        return QUESTIONS[pair]
    return ("Dono mein se kaunsa lagta hai?", {l1: l1, l2: l2})


st.set_page_config(page_title="Waste Segregation Agent", page_icon="♻️")
st.title("♻️ Waste Segregation Agent")
st.caption("Photo upload karo ya camera se click karo. Agent bataega ki kachra kis bin mein jayega.")


@st.cache_resource
def get_model():
    return load_model()


model, classes = get_model()

tab_upload, tab_camera = st.tabs(["📁 Upload photo", "📷 Use camera"])
with tab_upload:
    uploaded = st.file_uploader("Image chuno", type=["jpg", "jpeg", "png"])
with tab_camera:
    camera = st.camera_input("Item ki photo lo")

file = uploaded or camera

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Your item", width="stretch")

    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, 3)
    results = [(classes[i], p.item()) for p, i in zip(top.values, top.indices)]
    label, conf = results[0]

    with st.expander("Model ke top guesses"):
        for l, p in results:
            st.progress(float(p), text=f"{l}: {p:.0%}")

    final = label
    question = agent_decide(results)

    if question is None:
        st.success(f"Detected: **{label}** ({conf:.0%})")
    else:
        q, opts = question
        st.warning(
            f"Model ko pakka nahi hai (best guess: **{label}**, {conf:.0%}). "
            "Agent ek sawaal poochta hai:"
        )
        choice = st.radio(q, list(opts.keys()), index=None)
        if choice is None:
            st.stop()
        final = opts[choice]
        st.success(f"Agent ka final faisla: **{final}**")

    bin_name, tip = BIN_MAP[final]
    st.subheader(f"🗑️ {bin_name}")
    st.info(f"💡 {tip}")

    # Scan history mein save karo (ek hi photo + jawab sirf ek baar log hoga)
    key = (getattr(file, "file_id", file.name), final)
    if st.session_state.get("last_logged") != key:
        log_scan(label, conf, final, bin_name, question is not None)
        st.session_state["last_logged"] = key
