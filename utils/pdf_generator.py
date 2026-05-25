# ============================================================
# utils/pdf_generator.py
# Real-Time Mock Interview Evaluator
# ============================================================
# Generates a PDF report of the interview session.
# ============================================================

import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(grade: dict, emotion: dict, sentiment: dict, lexical: dict, 
                       content: dict, grammar: dict, video: dict, transcript: str) -> bytes:
    """Generate a PDF binary containing the full analysis report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    
    heading_style = styles['Heading2']
    heading_style.textColor = colors.HexColor("#080d1a")
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 14
    
    elements = []
    
    # Header
    elements.append(Paragraph("MockIQ Interview Evaluation Report", title_style))
    elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Overall Score
    elements.append(Paragraph(f"Overall Grade: {grade['letter_grade']} ({grade['score']:.1f}/100)", heading_style))
    elements.append(Paragraph(grade['summary'], normal_style))
    elements.append(Spacer(1, 15))
    
    # Metrics Table
    data = [
        ['Metric', 'Result', 'Details'],
        ['Tone', emotion.get('label', ''), f"{emotion.get('score', 0)}% confidence"],
        ['Sentiment', sentiment.get('label', ''), f"Polarity: {sentiment.get('polarity', 0):+.2f}"],
        ['Fluency', f"{lexical.get('fluency_score', 0):.0f}%", lexical.get('rating', '')],
        ['Grammar', f"{grammar.get('grammar_score', 0):.1f}%", f"Lexical Richness: {grammar.get('lexical_richness', 0):.2f}"],
        ['Video/Body', f"Smile: {video.get('smile_pct', 0)}%", f"Nervous: {video.get('nervous_pct', 0)}%"]
    ]
    
    table = Table(data, colWidths=[100, 150, 250])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#080d1a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f0f4ff")),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Content Feedback
    elements.append(Paragraph("Content Evaluation", heading_style))
    elements.append(Paragraph(content.get('content_feedback', 'N/A'), normal_style))
    elements.append(Spacer(1, 15))
    
    # Coaching
    elements.append(Paragraph("Coaching Feedback", heading_style))
    elements.append(Paragraph(f"<b>Tone:</b> {emotion.get('coaching', '')}", normal_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(f"<b>Fluency:</b> {lexical.get('coaching', '')}", normal_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(f"<b>Grammar:</b> {grammar.get('coaching', '')}", normal_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(f"<b>Body Language:</b> {video.get('coaching', '')}", normal_style))
    elements.append(Spacer(1, 20))
    
    # Transcript
    elements.append(Paragraph("Transcript", heading_style))
    elements.append(Paragraph(transcript, normal_style))
    
    # Build PDF
    doc.build(elements)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
