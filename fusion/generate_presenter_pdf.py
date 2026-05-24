"""
Generate TTM Group 42 Presenter's Script PDF using ReportLab.
Output: presenter_guide.pdf
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Colors ────────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0D2A4A")
BLUE   = colors.HexColor("#1565C0")
TEAL   = colors.HexColor("#00897B")
ORANGE = colors.HexColor("#E65100")
PURPLE = colors.HexColor("#6A1B9A")
RED    = colors.HexColor("#C62828")
GREEN  = colors.HexColor("#2E7D32")
LIGHT  = colors.HexColor("#F7F9FC")
MID    = colors.HexColor("#445566")
CARD_B = colors.HexColor("#E8F0FE")
CARD_T = colors.HexColor("#E0F2F1")
CARD_O = colors.HexColor("#FBE9E7")
CARD_P = colors.HexColor("#F3E5F5")
CARD_G = colors.HexColor("#E8F5E9")
WHITE  = colors.white

W, H = A4

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

base = S("base", fontName="Helvetica", fontSize=10, leading=14,
         textColor=MID, spaceAfter=4)

title_s = S("title_s", fontName="Helvetica-Bold", fontSize=22, leading=28,
            textColor=WHITE, alignment=TA_CENTER)

subtitle_s = S("subtitle_s", fontName="Helvetica", fontSize=13, leading=18,
               textColor=CARD_B, alignment=TA_CENTER)

cover_small = S("cover_small", fontName="Helvetica", fontSize=9, leading=13,
                textColor=colors.HexColor("#BBCFE0"), alignment=TA_CENTER)

section_s = S("section_s", fontName="Helvetica-Bold", fontSize=11, leading=15,
              textColor=WHITE)

slide_num_s = S("slide_num_s", fontName="Helvetica-Bold", fontSize=13, leading=16,
                textColor=NAVY)

presenter_s = S("presenter_s", fontName="Helvetica-Bold", fontSize=10, leading=14,
                textColor=BLUE)

body_s = S("body_s", fontName="Helvetica", fontSize=10, leading=15,
           textColor=MID, spaceAfter=3)

script_s = S("script_s", fontName="Helvetica-Oblique", fontSize=9.5, leading=14,
             textColor=colors.HexColor("#334455"), spaceAfter=3)

bold_s = S("bold_s", fontName="Helvetica-Bold", fontSize=10, leading=14,
           textColor=NAVY)

bullet_s = S("bullet_s", fontName="Helvetica", fontSize=9.5, leading=14,
             textColor=MID, leftIndent=14, spaceAfter=2,
             bulletFontName="Helvetica", bulletFontSize=9.5)

tip_s = S("tip_s", fontName="Helvetica-Oblique", fontSize=9, leading=13,
          textColor=PURPLE)

qa_q_s = S("qa_q_s", fontName="Helvetica-Bold", fontSize=9.5, leading=13,
           textColor=NAVY, spaceAfter=2)

qa_a_s = S("qa_a_s", fontName="Helvetica", fontSize=9, leading=13,
           textColor=MID, spaceAfter=6, leftIndent=10)

small_s = S("small_s", fontName="Helvetica", fontSize=8.5, leading=12,
            textColor=MID)

center_s = S("center_s", fontName="Helvetica", fontSize=9, leading=13,
             textColor=MID, alignment=TA_CENTER)

nav_s = S("nav_s", fontName="Helvetica", fontSize=8, leading=11,
          textColor=colors.HexColor("#8899AA"))


# ── Flowable helpers ──────────────────────────────────────────────────────────

class ColorBox(Flowable):
    """A filled colored rectangle spanning the full usable width."""
    def __init__(self, color, height=0.45*cm, content_para=None):
        self.color = color
        self.box_height = height
        self.content_para = content_para
        Flowable.__init__(self)

    def wrap(self, aW, aH):
        self.width = aW
        return aW, self.box_height

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.box_height, fill=1, stroke=0)


class LeftBorderBox(Flowable):
    """Box with a colored left border, light background, and content paragraphs."""
    def __init__(self, paragraphs, border_color, bg_color=None,
                 border_width=4, padding=8, radius=3):
        self.paragraphs = paragraphs
        self.border_color = border_color
        self.bg_color = bg_color or LIGHT
        self.border_width = border_width
        self.padding = padding
        self.radius = radius
        self._height = 0
        self._width = 0
        Flowable.__init__(self)

    def wrap(self, aW, aH):
        self._width = aW
        inner_w = aW - self.border_width - 2 * self.padding
        total_h = self.padding
        for p in self.paragraphs:
            w, h = p.wrap(inner_w, aH)
            total_h += h + 2
        total_h += self.padding
        self._height = total_h
        return aW, total_h

    def draw(self):
        c = self.canv
        # background
        c.setFillColor(self.bg_color)
        c.roundRect(0, 0, self._width, self._height, self.radius,
                    fill=1, stroke=0)
        # left border
        c.setFillColor(self.border_color)
        c.rect(0, 0, self.border_width, self._height, fill=1, stroke=0)
        # draw paragraphs
        x = self.border_width + self.padding
        y = self._height - self.padding
        inner_w = self._width - self.border_width - 2 * self.padding
        for p in self.paragraphs:
            w, h = p.wrap(inner_w, self._height)
            y -= h
            p.drawOn(c, x, y)
            y -= 2


def sp(n=0.2): return Spacer(1, n * cm)


def colored_header(text, bg=NAVY, fg=WHITE, size=12):
    """Returns a table row acting as a colored header."""
    style = ParagraphStyle("hdr", fontName="Helvetica-Bold", fontSize=size,
                           textColor=fg, leading=size + 4)
    p = Paragraph(text, style)
    t = Table([[p]], colWidths=[W - 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", [3, 3, 3, 3]),
    ]))
    return t


def slide_block(num, title, presenters, timing_str,
                keypoints, script_lines, tip=None):
    """Build one slide's worth of content as a list of flowables."""
    items = []

    # ── Header ──────────────────────────────────────────────────────────────
    items.append(sp(0.25))
    items.append(HRFlowable(width="100%", thickness=1, color=NAVY,
                             spaceAfter=4))

    # Slide label
    items.append(Paragraph(
        f"<b>SLIDE {num} — {title}</b>",
        S("sh", fontName="Helvetica-Bold", fontSize=13, leading=17,
          textColor=NAVY, spaceAfter=2)
    ))

    # Presenter box (blue left-border)
    items.append(LeftBorderBox(
        [Paragraph(f"<b>Presenter(s):</b> {presenters}", presenter_s)],
        border_color=BLUE, bg_color=CARD_B
    ))
    items.append(sp(0.12))

    # Timing box (orange left-border)
    items.append(LeftBorderBox(
        [Paragraph(f"<b>Suggested Time:</b> {timing_str}", body_s)],
        border_color=ORANGE, bg_color=CARD_O
    ))
    items.append(sp(0.12))

    # Key points
    items.append(Paragraph("<b>Key Points to Cover</b>", bold_s))
    items.append(sp(0.05))
    for kp in keypoints:
        items.append(Paragraph(f"• {kp}", bullet_s))
    items.append(sp(0.12))

    # Script
    items.append(Paragraph("<b>What to Say (Script)</b>", bold_s))
    items.append(sp(0.05))
    for line in script_lines:
        items.append(Paragraph(line, script_s))
    items.append(sp(0.1))

    if tip:
        items.append(LeftBorderBox(
            [Paragraph(f"<b>Presenter Tip:</b> {tip}", tip_s)],
            border_color=PURPLE, bg_color=CARD_P
        ))
        items.append(sp(0.1))

    return items


# ════════════════════════════════════════════════════════════════════════════
# Content data
# ════════════════════════════════════════════════════════════════════════════

SLIDES = [

    # ── SLIDE 1 ─────────────────────────────────────────────────────────────
    dict(
        num=1, title="Title / Introduction",
        presenters="Kanika Choudhary",
        timing="~30 seconds — Set the stage; do not go into detail yet.",
        keypoints=[
            "State the project name and team number clearly.",
            "Mention the dataset (Ego4D) and the exact task (TTM).",
            "Briefly preview the pipeline acronym so the audience knows what is coming.",
            "Cite the headline metric AUC = 0.847 to set expectations.",
        ],
        script=[
            "\"Good morning / afternoon everyone. We are <b>Group 42</b> from CS671, and our project "
            "is titled <i>'Who Is Talking To Me?'</i> — a system for Active Speaker Detection "
            "on egocentric video.",

            "The task we solve is called <b>Talking-to-Me, or TTM</b> for short. "
            "Given a short video clip from a wearable first-person camera, we want to automatically "
            "detect whether the person in front of the camera is speaking <i>directly</i> to the "
            "camera wearer.",

            "Our pipeline uses an <b>I3D video encoder</b>, a <b>ResNet-18 audio encoder</b>, "
            "fused through <b>cross-modal attention</b>, followed by a <b>Bidirectional LSTM</b> "
            "and <b>Bahdanau attention</b> before a final binary classifier.",

            "We achieved an <b>AUC-ROC of 0.847</b> and an <b>F1 score of 0.668</b> on the "
            "balanced test set, with only <b>3.68 million parameters</b>. "
            "I will now hand over to my teammates to walk you through each component in detail.\"",
        ],
        tip="Smile and make eye contact. Let the audience read the summary panel on the right "
            "before moving on. Don't rush this slide — it sets the tone."
    ),

    # ── SLIDE 2 ─────────────────────────────────────────────────────────────
    dict(
        num=2, title="TTM Problem Definition",
        presenters="Kanika Choudhary",
        timing="~60 seconds",
        keypoints=[
            "Clearly define TTM: binary classification from egocentric video.",
            "Distinguish TTM from general Active Speaker Detection (ASD).",
            "Explain why detecting speech alone is insufficient.",
            "Connect the task to real-world applications: AR/VR, smart assistants, social robots.",
        ],
        script=[
            "\"Let me first define the problem clearly.",

            "<b>Talking-to-Me, or TTM</b>, is a binary classification task. "
            "Given a short one-to-two second clip from an <b>egocentric</b> — that means a "
            "first-person wearable — camera, the goal is to predict whether the visible person "
            "is <b>talking directly to the camera wearer</b>. "
            "The label is: TTM equals 1 if they are addressing you, and 0 if they are not.",

            "Why is this hard? The person might be talking — but to someone <i>else</i> in the "
            "same room, not to you. Simply detecting speech is not enough. We need to reason about "
            "<b>who is being addressed</b>.",

            "This has very direct real-world applications. In <b>AR/VR headsets</b>, "
            "you want the system to identify which virtual agent the user is addressing. "
            "In <b>smart voice assistants</b>, the device should respond only when the user "
            "is directly talking to it. In <b>social robots</b>, robots need to know who "
            "is addressing them in a multi-person scene.",

            "The key insight is: we need <b>both audio and visual cues together</b>. "
            "Lips, face orientation, and the sound of speech must be jointly analyzed.\"",
        ],
        tip="Point to the three application boxes on the right (AR/VR, Smart Assistants, "
            "Social Robotics) as you mention each one. Pause briefly after each."
    ),

    # ── SLIDE 3 ─────────────────────────────────────────────────────────────
    dict(
        num=3, title="Motivation & Challenges",
        presenters="Aman Sharma",
        timing="~60 seconds",
        keypoints=[
            "Walk through each challenge box: multiple speakers, occlusions, lip ambiguity, "
            "class imbalance, pre-training domain gap.",
            "Emphasize class imbalance — this is why AUC/F1 matter more than accuracy.",
            "Explain why audio-only or video-only systems fail.",
            "Lead naturally into: 'therefore we need multimodal fusion'.",
        ],
        script=[
            "\"Before we dive into our solution, let me explain why this problem is genuinely difficult.",

            "<b>First, multiple speakers.</b> In real social settings, three to five people may "
            "talk simultaneously. An audio-only system hears all of them and cannot determine "
            "who is addressing the camera wearer.",

            "<b>Second, occlusions and camera motion.</b> Because the camera moves with the wearer, "
            "faces can blur, disappear, or appear at unusual angles, making video analysis "
            "unreliable on its own.",

            "<b>Third, lip movement ambiguity.</b> Someone might move their lips while laughing, "
            "whispering, or mouthing words silently — so lip motion alone does not prove active speech.",

            "<b>Fourth, class imbalance.</b> Only a small fraction of clips are TTM-positive. "
            "A naive model that always predicts 'not talking to me' would get high accuracy but "
            "completely fail at the actual task. That is why we use <b>AUC-ROC and F1</b> "
            "as our primary metrics, not raw accuracy.",

            "<b>Finally, pre-training domain gap.</b> Our backbones are pretrained on Kinetics-400 "
            "action videos, which looks very different from close-up social face interactions "
            "in Ego4D.",

            "<b>The conclusion:</b> no single modality is sufficient. Robust TTM detection "
            "requires multimodal fusion with temporal context — exactly what our model does.\"",
        ],
        tip=None
    ),

    # ── SLIDE 4 ─────────────────────────────────────────────────────────────
    dict(
        num=4, title="Dataset Overview",
        presenters="Naman",
        timing="~60 seconds",
        keypoints=[
            "Introduce Ego4D: large-scale egocentric daily-life video.",
            "Explain the subset: 70k training clips, 6,390 test clips.",
            "Explain each evaluation metric and why it matters.",
            "Preview the precision-recall tradeoff so the results slide makes sense.",
        ],
        script=[
            "\"Our dataset is the <b>Ego4D Social Benchmark</b>. Ego4D contains over "
            "<b>700 hours</b> of egocentric video from daily life: people cooking, socializing, "
            "playing sports, and having conversations.",

            "For the TTM task, each clip is annotated with a binary label. "
            "Video is at 30 FPS, 1080p resolution, and audio is at 16 kHz.",

            "We used a balanced subset of <b>70,000 training clips</b>. "
            "The official <b>test set has 6,390 clips</b>.",

            "<b>AUC-ROC</b> measures how well the model separates positives from negatives "
            "across all possible thresholds. AUC = 0.5 is random, 1.0 is perfect. "
            "<b>We achieved 0.847.</b>",

            "<b>F1 Score</b> balances precision and recall for the TTM-positive class — "
            "critical because of class imbalance. <b>Our F1 is 0.668.</b>",

            "<b>Precision = 0.857</b>: of all clips predicted as TTM-positive, 85.7% were "
            "actually positive — very few false alarms.",

            "<b>Recall = 0.547</b>: we caught 54.7% of true TTM events. "
            "This is an area for future improvement.\"",
        ],
        tip=None
    ),

    # ── SLIDE 5 ─────────────────────────────────────────────────────────────
    dict(
        num=5, title="Complete Pipeline Architecture",
        presenters="Mihir Chandra",
        timing="~90 seconds",
        keypoints=[
            "Walk through the diagram left-to-right.",
            "Explain the two parallel input streams: video (top) and audio (bottom).",
            "Explain where they merge: Cross-Modal Attention.",
            "Trace the temporal path: BiLSTM → Bahdanau → FC → Output.",
            "Mention the final numbers: AUC 0.847, 3.68M params.",
        ],
        script=[
            "\"Let me give you the <b>bird's-eye view</b> of our full pipeline before my "
            "teammates explain each block in depth.",

            "Looking at the architecture diagram, you see <b>two parallel streams</b>.",

            "The <b>top stream handles video</b>. We take the input clip, detect the face with "
            "RetinaFace, then pass face crops through <b>I3D-R50</b> — a 3D CNN pretrained on "
            "Kinetics-400. Output: a 512-dimensional video embedding per time step.",

            "The <b>bottom stream handles audio</b>. We take raw 16 kHz audio, convert it to a "
            "<b>Mel spectrogram</b>, then pass it through <b>ResNet-18</b> pretrained on ImageNet. "
            "Output: a matching 512-dimensional audio embedding per time step.",

            "These two streams meet at <b>Cross-Modal Attention</b>. Video features attend to "
            "audio features and vice versa, so the model learns which audio is relevant to the "
            "visual content.",

            "The fused features — one per clip — are stacked into a sequence of <b>T = 16</b> "
            "time steps and fed into a <b>2-layer Bidirectional LSTM</b>.",

            "The BiLSTM output goes through <b>Bahdanau Attention</b>, which assigns importance "
            "weights to each time step, so the classifier focuses on the most informative moments.",

            "Finally, a <b>fully-connected layer with sigmoid</b> produces the TTM probability, "
            "thresholded at 0.139 for the binary prediction.",

            "The whole model has only <b>3.68 million parameters</b> and achieves "
            "<b>AUC-ROC = 0.847</b>.\"",
        ],
        tip="Use a laser pointer to trace arrows from left to right as you describe each block. "
            "Let the audience follow the data flow visually."
    ),

    # ── SLIDE 6 ─────────────────────────────────────────────────────────────
    dict(
        num=6, title="Video Processing Pipeline",
        presenters="Mihir Chandra (first ~45 s) + Harshit (second ~45 s)",
        timing="~90 seconds total",
        keypoints=[
            "Explain why 2D CNNs miss temporal motion — they process frames independently.",
            "Explain the SlowFast dual-pathway design: Slow = semantic, Fast = motion.",
            "Clarify: I3D is used in the fusion model; SlowFast was tested as a standalone baseline.",
            "Mention the overfitting problem and the fix: frozen early layers + dropout.",
        ],
        script=[
            "<b>Mihir:</b> \"To process video we need to capture not just what a face looks like, "
            "but how it <i>moves</i> over time. Standard 2D CNNs process each frame independently "
            "and miss motion. Instead we use <b>3D Convolutional Neural Networks</b> that "
            "convolve across both space and time, learning lip motion and head movement patterns.",

            "For baseline experiments we tested <b>SlowFast-R50</b> — a dual-pathway architecture. "
            "The <b>Slow pathway</b> processes at 4 FPS with high spatial resolution, capturing "
            "detailed facial appearance. The <b>Fast pathway</b> processes at 32 FPS with a "
            "lightweight network, capturing rapid motion like jaw and lip dynamics. "
            "The two pathways share information via lateral connections.",

            "Both are pretrained on <b>Kinetics-400</b>.\"",

            "<b>Harshit:</b> \"In our final fusion model, we use <b>I3D-R50</b> as the video "
            "encoder — also spatiotemporal, but with a simpler architecture that integrates "
            "cleanly with the rest of our pipeline.",

            "An important observation: the video-only model <b>overfit severely</b>. "
            "Training accuracy hit near 100% while validation plateaued. "
            "Our fix: freeze early backbone layers, fine-tune only the final blocks, "
            "add dropout and weight decay.",

            "The video encoder produces a <b>512-dim embedding</b> per clip that flows into "
            "cross-modal attention.\"",
        ],
        tip=None
    ),

    # ── SLIDE 7 ─────────────────────────────────────────────────────────────
    dict(
        num=7, title="Audio Processing Pipeline",
        presenters="Naman (Steps 1–2) + Sowmika Rao (Steps 3–4)",
        timing="~90 seconds total",
        keypoints=[
            "Explain why raw audio is not fed directly: CNNs need 2D image-like input.",
            "Explain what a Mel spectrogram is and why the Mel scale models human perception.",
            "Explain how ResNet-18 treats the spectrogram as a single-channel image.",
            "State the audio-only accuracy: 66.78% — useful baseline reference.",
        ],
        script=[
            "<b>Naman:</b> \"Let me walk you through the audio processing pipeline.",

            "<b>Step 1 — Audio Extraction.</b> We extract raw 16 kHz mono audio and segment it "
            "into 1-second windows time-aligned with video clips.",

            "<b>Step 2 — Mel Spectrogram.</b> A raw audio waveform is a 1D signal. "
            "CNNs work best with 2D image-like inputs. We apply a <b>Short-Time Fourier "
            "Transform</b> to get frequency vs. time, then map frequencies onto the "
            "<b>Mel scale</b> — a perceptual scale that matches how humans hear. "
            "Lower frequencies are spread out; higher ones compressed. "
            "The result is a <b>1 × 64 × 50</b> spectrogram.\"",

            "<b>Sowmika:</b> \"<b>Step 3 — ResNet-18 Audio CNN.</b> We treat the spectrogram "
            "as a single-channel image and pass it through <b>ResNet-18</b> initialized from "
            "ImageNet weights. The CNN learns spectral patterns associated with active speech.",

            "Audio-only, this model achieves <b>66.78% validation accuracy</b> — a solid "
            "baseline, but clearly insufficient on its own.",

            "<b>Step 4 — Audio Embedding.</b> Global Average Pooling produces a "
            "<b>512-dim embedding</b> per clip, matching the video embedding dimension — "
            "essential for the cross-modal attention fusion step.\"",
        ],
        tip=None
    ),

    # ── SLIDE 8 ─────────────────────────────────────────────────────────────
    dict(
        num=8, title="Temporal Modeling with BiLSTM",
        presenters="Vikky Kumar (first 40 s) + Vershita Yadav (second 35 s)",
        timing="~75 seconds",
        keypoints=[
            "Explain why single-clip decisions are noisy and insufficient.",
            "Explain BiLSTM intuitively: forward (past context) + backward (future context).",
            "State the configuration: T=16 clips, 2 layers, hidden=256 per direction.",
            "Mention dropout = 0.5 as a deliberate regularization choice.",
        ],
        script=[
            "<b>Vikky:</b> \"After cross-modal attention produces a fused feature per clip, "
            "should we make a binary decision on just <i>one clip</i> at a time? No — "
            "speech is a continuous, evolving behavior. Whether someone is talking to you "
            "unfolds over several seconds.",

            "So we <b>stack T = 16 consecutive clips into a sequence</b> and model it with a "
            "<b>Bidirectional LSTM</b>. An LSTM is a recurrent network designed for sequential "
            "data. The <b>bidirectional</b> variant processes forward — using past context — "
            "and backward — using future context. "
            "Sometimes a trailing word at the end of the sequence is the most informative cue.\"",

            "<b>Vershita:</b> \"Our BiLSTM has <b>2 stacked layers, hidden size 256 per "
            "direction</b> — 512 total per time step.",

            "During training we use a sequence stride of 4 clips to create more training "
            "examples. During inference we use stride 8 for efficiency.",

            "We apply <b>dropout of 0.5</b> between layers to prevent the severe overfitting "
            "we observed in the video backbone.",

            "The BiLSTM outputs 16 hidden states — one per clip — which feed into "
            "Bahdanau attention.\"",
        ],
        tip=None
    ),

    # ── SLIDE 9 ─────────────────────────────────────────────────────────────
    dict(
        num=9, title="Bahdanau Attention Mechanism",
        presenters="Vershita Yadav",
        timing="~60 seconds",
        keypoints=[
            "Explain the problem with mean pooling: treats all clips equally.",
            "Explain attention weights: α_t tells us which clips matter most.",
            "Walk through the three equations in plain language.",
            "Point to the bar chart: high attention at clips 7–9 where lips + audio align.",
        ],
        script=[
            "\"After BiLSTM gives us 16 hidden states, we need to <b>combine them into "
            "one vector</b> for the final classifier.",

            "The simplest approach is <b>mean pooling</b> — average all 16 states equally. "
            "But this is poor: some clips have the person looking away, others have perfect "
            "lip-audio synchronization. We should focus on the <b>informative clips</b>.",

            "<b>Bahdanau Attention</b> assigns a weight α_t to each time step t, "
            "where all weights sum to 1. The final context vector c is the weighted sum "
            "of all hidden states.",

            "The math is simple: compute an energy score e_t = v·tanh(Wh_t + b) "
            "for each hidden state, normalize with softmax to get α_t, "
            "then c = Σ α_t h_t.",

            "Look at the bar chart — the model assigns <b>high attention to clips 7–9</b> "
            "where audio and visual signals are most synchronized, and low attention to "
            "the beginning and end of the sequence.",

            "This improves accuracy <i>and</i> gives us <b>interpretability</b> — "
            "we can inspect which clips drove the decision.\"",
        ],
        tip=None
    ),

    # ── SLIDE 10 ────────────────────────────────────────────────────────────
    dict(
        num=10, title="Training Strategy",
        presenters="Aman Sharma (first 35 s) + Vershita Yadav (second 25 s)",
        timing="~60 seconds",
        keypoints=[
            "Weighted BCE loss: higher penalty for TTM-positive misclassifications.",
            "Weighted random sampler: oversample positives so both classes appear per batch.",
            "AdamW + ReduceLROnPlateau: adaptive learning rate.",
            "Early stopping at epoch 9: validation AP plateaued while train loss kept falling.",
            "Threshold = 0.139: tuned to maximize F1, not accuracy.",
        ],
        script=[
            "<b>Aman:</b> \"We use <b>Weighted Binary Cross-Entropy loss</b>. "
            "Because TTM-positive clips are rare, standard BCE is dominated by negatives. "
            "We assign a higher loss weight to positive samples to compensate.",

            "We also use a <b>weighted random sampler</b> that oversamples positives so both "
            "classes appear roughly equally per batch. Batch size is <b>256 sequences</b>.",

            "Our optimizer is <b>AdamW with learning rate 3×10⁻⁴</b>, with "
            "<b>ReduceLROnPlateau</b> to lower the LR when validation stops improving. "
            "Dropout is 0.5 throughout.\"",

            "<b>Vershita:</b> \"Looking at the training curves: train loss decreases "
            "steadily over all 18 epochs, but <b>validation AP peaks at epoch 9</b> and "
            "oscillates without improvement — a classic overfitting pattern. "
            "We apply <b>early stopping</b> and use the epoch-9 checkpoint.",

            "Instead of the default 0.5 threshold, we tuned to <b>threshold = 0.139</b>. "
            "This lowers the bar for predicting TTM=1, improving recall without "
            "sacrificing too much precision.\"",
        ],
        tip=None
    ),

    # ── SLIDE 11 ────────────────────────────────────────────────────────────
    dict(
        num=11, title="Experimental Results",
        presenters="Kanika Choudhary",
        timing="~75 seconds",
        keypoints=[
            "Read the four headline numbers: AUC 0.847, F1 0.668, Precision 0.857, Recall 0.547.",
            "Explain the confusion matrix in plain language: TN, TP, FP, FN.",
            "Show the component comparison: Full Fusion > Audio-only > SlowFast-only.",
            "Explain why the validation numbers (AUC 0.995, Acc 98.2%) are inflated.",
        ],
        script=[
            "\"Let us now look at the actual results on the <b>balanced test set of 6,390 clips</b>.",

            "Our headline numbers: <b>AUC-ROC = 0.847</b>, <b>F1 = 0.668</b>, "
            "<b>Precision = 0.857</b>, <b>Recall = 0.547</b>.",

            "The <b>confusion matrix</b>: we correctly labelled 5,807 non-TTM clips as negative "
            "(true negatives) and 3,497 TTM clips as positive (true positives). "
            "We had 583 false alarms and 2,893 missed detections. "
            "High precision means our positive predictions are very reliable. "
            "Moderate recall means we still miss some true events.",

            "The <b>component comparison</b> is telling: SlowFast-only achieves 61.9% accuracy, "
            "audio CNN alone achieves 66.8%, and the <b>full fusion model reaches 72.8%</b>. "
            "Both modalities are essential.",

            "One note: the validation set shows AUC = 0.995 and accuracy = 98.2%. "
            "This is <b>misleading</b> — the validation split is highly imbalanced with far "
            "more negatives than positives. The balanced test set is the correct benchmark.\"",
        ],
        tip=None
    ),

    # ── SLIDE 12 ────────────────────────────────────────────────────────────
    dict(
        num=12, title="Comparison with Prior Work",
        presenters="Aman Sharma",
        timing="~60 seconds",
        keypoints=[
            "Acknowledge that TalkNet/ASDNet/MAAS/LightASD use AVA-ASD — different dataset.",
            "Highlight unique contribution: egocentric, two-stage attention design.",
            "Emphasize parameter efficiency: 3.68M vs 15–50M for prior work.",
            "Compare directly against the Ego4D baseline: ~60% → 72.8%.",
        ],
        script=[
            "\"Let me place our results in context. The table compares us with four published methods.",

            "<b>TalkNet, ASDNet, MAAS,</b> and <b>Light-ASD</b> all evaluate on "
            "<b>AVA-ASD</b> — a third-person movie dataset. They report mean Average Precision. "
            "Direct numerical comparison is not perfectly fair, but it shows the architectural "
            "landscape. They achieve mAP 88–94% with 1 to 50 million parameters.",

            "The <b>Ego4D baseline</b> from Grauman et al. is the most directly comparable: "
            "~60% accuracy, ~11M parameters. "
            "<b>We achieve 72.8% accuracy with only 3.68M parameters</b>.",

            "What makes us unique? <b>First</b>, we are the only egocentric model with a "
            "<b>two-stage attention design</b> — cross-modal attention for fusion, "
            "Bahdanau attention for temporal focus. <b>Second</b>, we explicitly address "
            "<b>class imbalance</b> through weighted BCE, weighted sampling, and threshold "
            "tuning — which prior ASD methods do not tackle. <b>Third</b>, we use "
            "4–13× fewer parameters than TalkNet, ASDNet, and MAAS.\"",
        ],
        tip=None
    ),

    # ── SLIDE 13 ────────────────────────────────────────────────────────────
    dict(
        num=13, title="Risks & Mitigation Strategies",
        presenters="Harshit (first 25 s) + Sowmika Rao (second 20 s)",
        timing="~45 seconds — Move through briskly.",
        keypoints=[
            "These are real challenges encountered, not theoretical.",
            "Cover: OOM → FP16 + grad accumulation.",
            "Video overfitting → frozen layers + dropout + early stopping.",
            "Gradient vanishing → grad clipping + attention shortcuts.",
            "Class imbalance → weighted sampling + F1/AUC monitoring.",
            "A-V sync offsets → preprocessing checks + temporal augmentation.",
        ],
        script=[
            "<b>Harshit:</b> \"During development we hit several real engineering challenges.",

            "<b>GPU out-of-memory</b>: combining 3D CNN + BiLSTM is heavy. Fix: "
            "<b>FP16 mixed-precision training</b>, gradient accumulation, reduced resolution.",

            "<b>Video CNN overfitting</b>: training accuracy saturated quickly. Fix: "
            "<b>dropout, weight decay, frozen early layers, early stopping at epoch 9</b>.",

            "<b>Gradient vanishing</b>: LSTM gradients weaken over long sequences. Fix: "
            "<b>gradient clipping</b> and Bahdanau attention shortcut connections.\"",

            "<b>Sowmika:</b> \"<b>Class imbalance</b>: TTM-positive clips are rare. Fix: "
            "<b>weighted sampling, weighted BCE loss</b>, and monitoring F1/AUC instead of "
            "raw accuracy.",

            "<b>Audio-visual sync offsets</b>: small timing mismatches in Ego4D recordings. "
            "Fix: <b>synchronization preprocessing checks</b> and temporal augmentation to "
            "make the model robust to small A-V shifts.\"",
        ],
        tip=None
    ),

    # ── SLIDE 14 ────────────────────────────────────────────────────────────
    dict(
        num=14, title="Team Role Distribution",
        presenters="Kanika Choudhary",
        timing="~30 seconds — Keep brief, this is a courtesy acknowledgment slide.",
        keypoints=[
            "Briefly credit each sub-team's module.",
            "Show the modular pipeline structure enabled parallel development.",
        ],
        script=[
            "\"Our team of eight worked in sub-groups, each owning a specific pipeline module.",

            "<b>Mihir and Harshit</b> — 3D video CNN using I3D and SlowFast.",
            "<b>Aman and Kanika</b> — video preprocessing and face detection.",
            "<b>Naman and Sowmika</b> — audio pipeline with Mel spectrogram and ResNet-18.",
            "<b>Vikky and Vershita</b> — BiLSTM, both attention mechanisms, and the "
            "end-to-end training loop.",

            "The modular design allowed parallel development and easy ablation testing "
            "of individual components.\"",
        ],
        tip=None
    ),

    # ── SLIDE 15 ────────────────────────────────────────────────────────────
    dict(
        num=15, title="Conclusion & Future Work",
        presenters="Vikky Kumar",
        timing="~60 seconds",
        keypoints=[
            "Summarize the four main findings: multimodal fusion, cross-modal attention, "
            "BiLSTM, precision-recall tradeoff.",
            "Present four concrete future directions.",
            "End with a forward-looking statement.",
        ],
        script=[
            "\"Let me summarize what we have demonstrated.",

            "<b>First:</b> multimodal fusion works. Video-only and audio-only models both "
            "underperformed. Combining them gave a clear jump in AUC and accuracy.",

            "<b>Second:</b> cross-modal attention is a powerful fusion mechanism. "
            "Letting video features attend to audio and vice versa learns meaningful "
            "inter-modal correlations before temporal modeling.",

            "<b>Third:</b> temporal context via BiLSTM is essential. "
            "Single-clip decisions are noisy. A 16-clip sequence allows the model to "
            "accumulate evidence over time.",

            "<b>Fourth:</b> precision vs. recall is a fundamental tradeoff. "
            "Our model is precise — positive predictions are mostly correct — but it misses "
            "some true events. Threshold tuning and better training can shift this balance.",

            "For <b>future work</b>: (1) <b>Video Transformers</b> like Video Swin or "
            "TimeSformer for more powerful temporal modeling; "
            "(2) <b>Adaptive thresholding</b> and probability calibration to improve recall; "
            "(3) <b>Real-time edge deployment</b> for AR/VR; "
            "(4) <b>Multi-speaker attribution</b> — predicting TTM for multiple visible "
            "faces simultaneously.",

            "We believe this is a strong, parameter-efficient approach to egocentric "
            "active speaker detection.\"",
        ],
        tip=None
    ),

    # ── SLIDE 16 ────────────────────────────────────────────────────────────
    dict(
        num=16, title="Thank You / Q&A",
        presenters="All Members",
        timing="~30 seconds then open Q&A",
        keypoints=[
            "Brief closing statement re-stating key results.",
            "All members should stand or be present for Q&A.",
            "See Q&A preparation section below for likely questions.",
        ],
        script=[
            "\"Thank you for your attention. We are happy to take questions.",

            "To recap: our Group 42 TTM system uses <b>I3D video features + ResNet-18 "
            "audio features</b>, fused through <b>cross-modal attention</b>, modeled "
            "temporally by a <b>2-layer BiLSTM</b>, focused by <b>Bahdanau attention</b>, "
            "and classified with a sigmoid output.",

            "We achieve <b>AUC-ROC = 0.847</b>, <b>F1 = 0.668</b>, "
            "<b>Precision = 0.857</b>, <b>Recall = 0.547</b> on the balanced Ego4D test "
            "set with only <b>3.68 million parameters</b>.\"",
        ],
        tip=None
    ),
]

# ── Q&A DATA ─────────────────────────────────────────────────────────────────

QA = [
    ("Why use a threshold of 0.139 instead of 0.5?",
     "The default threshold of 0.5 is optimal only when the class distribution is balanced. "
     "Because TTM-positive clips are rare, the model's sigmoid outputs are systematically "
     "low. Tuning the threshold on the validation set to 0.139 maximizes F1 by accepting "
     "lower-confidence positive predictions."),

    ("Why not use a Transformer instead of a BiLSTM?",
     "BiLSTMs are more memory-efficient and train faster with limited GPU resources. "
     "They are also a proven baseline for temporal sequence modeling. Transformers are a "
     "natural next step but require significantly more compute and data."),

    ("Why is the validation AUC 0.995 but test AUC only 0.847?",
     "The validation split is heavily imbalanced — far more negatives than positives. "
     "A model good at rejecting negatives scores very high on imbalanced AUC. "
     "The balanced test set gives a fairer and more meaningful evaluation."),

    ("What is cross-modal attention and why is it better than simple concatenation?",
     "Concatenation blindly merges two feature vectors with no selectivity. "
     "Cross-modal attention lets each modality query the other: audio features help "
     "determine which video regions to focus on, and video features help determine "
     "which audio frequencies are relevant. This selective fusion is more expressive."),

    ("What is the biggest limitation of your system?",
     "Recall of 0.547 means we miss about half of true TTM events. "
     "This is partly due to class imbalance in training, and partly because some "
     "TTM events are genuinely ambiguous. Better temporal windows and stronger data "
     "augmentation could help."),

    ("How does your system compare to the official Ego4D TTM leaderboard?",
     "We substantially improve over the Ego4D baseline (~60% accuracy, ~11M params) "
     "achieving 72.8% accuracy with only 3.68M parameters. Direct leaderboard comparison "
     "requires submitting to the Ego4D evaluation server, which is future work."),
]


# ════════════════════════════════════════════════════════════════════════════
# PDF BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_pdf(out_path):

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2 * cm,
        title="Group 42 CS671 TTM — Presenter's Script",
        author="Group 42, IIT, CS671",
    )

    story = []

    # ── COVER ────────────────────────────────────────────────────────────────
    # Navy background via a big colored table
    cover_data = [[
        Paragraph(
            "<br/><br/><br/>"
            "<font size=24><b>PRESENTER'S SCRIPT</b></font><br/><br/>"
            "<font size=14 color='#A8C4E0'>&#8220;Who Is Talking To Me?&#8221;</font><br/>"
            "<font size=11 color='#8aaabf'>TTM Active Speaker Detection on Ego4D</font><br/><br/>"
            "<font size=10 color='#E65100'>————————————————————————————</font><br/><br/>"
            "<font size=11><b>Group 42 · Indian Institute of Technology · CS671</b></font><br/><br/>"
            "<font size=9 color='#9DB8CC'>"
            "Kanika Choudhary · Aman Sharma · Vikky Kumar · Vershita Yadav<br/>"
            "Mihir Chandra · Harshit · Sowmika Rao · Naman"
            "</font><br/><br/>"
            "<font size=10 color='#E65100'>————————————————————————————</font><br/><br/>"
            "<font size=12><b>Total Presentation Time: ~15 Minutes</b></font><br/>"
            "<font size=9 color='#9DB8CC'>"
            "16 Slides · Speaker Notes Per Slide · Role-Assigned Presenters"
            "</font><br/><br/><br/>"
            "<font size=9 color='#7799BB'>"
            "AUC-ROC = 0.847  ·  F1 = 0.668  ·  Precision = 0.857  ·  "
            "Recall = 0.547  ·  Accuracy = 72.8%  ·  Params = 3.68M"
            "</font><br/><br/>",
            ParagraphStyle("cov", fontName="Helvetica", fontSize=11,
                           textColor=WHITE, alignment=TA_CENTER, leading=16)
        )
    ]]
    cover_tbl = Table(cover_data, colWidths=[W - 4 * cm])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 30),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 30),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story.append(cover_tbl)
    story.append(PageBreak())

    # ── OVERVIEW TABLE ───────────────────────────────────────────────────────
    story.append(Paragraph(
        "Presentation Overview & Timing Guide",
        S("ov", fontName="Helvetica-Bold", fontSize=15, textColor=NAVY,
          leading=20, spaceAfter=8)
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=8))

    overview_rows = [
        [Paragraph("<b>Slide</b>", S("th", fontName="Helvetica-Bold", fontSize=9,
                                     textColor=WHITE, leading=12)),
         Paragraph("<b>Title</b>", S("th", fontName="Helvetica-Bold", fontSize=9,
                                     textColor=WHITE, leading=12)),
         Paragraph("<b>Presenter(s)</b>", S("th", fontName="Helvetica-Bold", fontSize=9,
                                             textColor=WHITE, leading=12)),
         Paragraph("<b>Time</b>", S("th", fontName="Helvetica-Bold", fontSize=9,
                                    textColor=WHITE, leading=12))],
    ]
    rows_data = [
        (1, "Title / Introduction", "Kanika Choudhary", "00:30"),
        (2, "TTM Problem Definition", "Kanika Choudhary", "01:00"),
        (3, "Motivation & Challenges", "Aman Sharma", "01:00"),
        (4, "Dataset Overview", "Naman", "01:00"),
        (5, "Complete Pipeline Architecture", "Mihir Chandra", "01:30"),
        (6, "Video Processing Pipeline", "Mihir Chandra + Harshit", "01:30"),
        (7, "Audio Processing Pipeline", "Naman + Sowmika Rao", "01:30"),
        (8, "BiLSTM Temporal Modeling", "Vikky Kumar + Vershita Yadav", "01:15"),
        (9, "Bahdanau Attention Mechanism", "Vershita Yadav", "01:00"),
        (10, "Training Strategy", "Aman Sharma + Vershita Yadav", "01:00"),
        (11, "Experimental Results", "Kanika Choudhary", "01:15"),
        (12, "Comparison with Prior Work", "Aman Sharma", "01:00"),
        (13, "Risks & Mitigations", "Harshit + Sowmika Rao", "00:45"),
        (14, "Team Role Distribution", "Kanika Choudhary", "00:30"),
        (15, "Conclusion & Future Work", "Vikky Kumar", "01:00"),
        (16, "Thank You / Q&A", "All Members", "00:30"),
        ("", "TOTAL", "", "~15:15"),
    ]
    cell_s = ParagraphStyle("cell", fontName="Helvetica", fontSize=9,
                             textColor=MID, leading=12)
    bold_cell = ParagraphStyle("bcell", fontName="Helvetica-Bold", fontSize=9,
                                textColor=NAVY, leading=12)
    for i, (n, t, p, ti) in enumerate(rows_data):
        bg = CARD_B if i % 2 == 0 else WHITE
        is_total = (n == "")
        style = bold_cell if is_total else cell_s
        overview_rows.append([
            Paragraph(str(n), style),
            Paragraph(f"<b>{t}</b>" if is_total else t, style),
            Paragraph(p, style),
            Paragraph(f"<b>{ti}</b>" if is_total else ti, style),
        ])

    ov_tbl = Table(overview_rows, colWidths=[1.2 * cm, 6 * cm, 5.5 * cm, 1.8 * cm])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCDDEE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [CARD_B, WHITE]),
        ("BACKGROUND", (0, -1), (-1, -1), NAVY),
        ("TEXTCOLOR", (0, -1), (-1, -1), WHITE),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]
    ov_tbl.setStyle(TableStyle(style_cmds))
    story.append(ov_tbl)
    story.append(sp(0.4))

    # Quick checklist
    story.append(LeftBorderBox(
        [Paragraph("<b>Before You Begin — Quick Checklist</b>", bold_s),
         Paragraph("• Advance to your slide <i>before</i> you start speaking.", bullet_s),
         Paragraph("• Speak slowly and clearly — the audience may not know ML jargon.", bullet_s),
         Paragraph("• Use the pointer to highlight diagrams as you describe them.", bullet_s),
         Paragraph("• Each presenter should stand or be visible when speaking.", bullet_s),
         Paragraph("• Q&A: keep all group members on stage — answer collectively if unsure.", bullet_s),
         ],
        border_color=ORANGE, bg_color=CARD_O
    ))
    story.append(PageBreak())

    # ── SLIDE SECTIONS ────────────────────────────────────────────────────────
    story.append(Paragraph(
        "Slide-by-Slide Speaker Notes",
        S("sec_hdr", fontName="Helvetica-Bold", fontSize=15, textColor=NAVY,
          leading=20, spaceAfter=4)
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=6))

    for slide in SLIDES:
        elems = slide_block(
            num=slide["num"],
            title=slide["title"],
            presenters=slide["presenters"],
            timing_str=slide["timing"],
            keypoints=slide["keypoints"],
            script_lines=slide["script"],
            tip=slide.get("tip"),
        )
        story.extend(elems)

    story.append(PageBreak())

    # ── Q&A SECTION ───────────────────────────────────────────────────────────
    story.append(Paragraph(
        "Likely Q&A Questions — Prepare These Answers",
        S("qa_hdr", fontName="Helvetica-Bold", fontSize=14, textColor=NAVY,
          leading=18, spaceAfter=4)
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=8))
    story.append(LeftBorderBox(
        [Paragraph(
            "All members should be prepared to answer any of the following. "
            "If unsure, it is acceptable to say: "
            "<i>'We explored that direction but focused on X due to Y - "
            "great point for future work.'</i>",
            body_s)],
        border_color=BLUE, bg_color=CARD_B
    ))
    story.append(sp(0.2))

    for i, (q, a) in enumerate(QA, 1):
        story.append(Paragraph(f"<b>Q{i}. {q}</b>", qa_q_s))
        story.append(Paragraph(a, qa_a_s))

    story.append(sp(0.5))

    # ── FOOTER ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=8))
    story.append(Paragraph(
        "— END OF PRESENTER'S SCRIPT —",
        S("end", fontName="Helvetica-Bold", fontSize=12, textColor=NAVY,
          alignment=TA_CENTER, leading=16, spaceAfter=4)
    ))
    story.append(Paragraph(
        "Group 42  ·  CS671  ·  IIT  ·  2026",
        S("end2", fontName="Helvetica", fontSize=9, textColor=MID,
          alignment=TA_CENTER, leading=13)
    ))

    # ── Page numbering ────────────────────────────────────────────────────────
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(doc.leftMargin, H - 1.6 * cm,
                    W - doc.leftMargin - doc.rightMargin, 0.55 * cm,
                    fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.drawString(doc.leftMargin + 0.2 * cm, H - 1.25 * cm,
                          "Group 42  ·  CS671  ·  IIT  ·  TTM Presenter's Script")
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(W - doc.rightMargin - 0.2 * cm,
                               H - 1.25 * cm, f"Page {doc.page}")
        canvas.setFillColor(ORANGE)
        canvas.rect(doc.leftMargin, doc.bottomMargin - 0.35 * cm,
                    W - doc.leftMargin - doc.rightMargin, 0.18 * cm,
                    fill=1, stroke=0)
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF generated: {out_path}")


if __name__ == "__main__":
    build_pdf("/usershome/cs671_user13/group_42/fusion/presenter_guide.pdf")
