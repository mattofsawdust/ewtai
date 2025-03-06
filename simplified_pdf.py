"""
Simplified PDF generation module for the EWTai app.
This creates simple PDFs with tables instead of charts.
"""

import os
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def create_simple_pdf(symbol, data, summary=None, prediction=None, analysis_type="Basic"):
    """Create a simple PDF report without complex charts."""
    try:
        # Create results directory if needed
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(results_dir, f"{symbol}_{analysis_type.lower().replace(' ', '_')}_{timestamp}.pdf")
        
        # Create a PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Custom styles
        bullish_style = ParagraphStyle(
            'Bullish',
            parent=normal_style,
            textColor=colors.green
        )
        bearish_style = ParagraphStyle(
            'Bearish',
            parent=normal_style,
            textColor=colors.red
        )
        
        # Elements to add to the PDF
        elements = []
        
        # Title
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        elements.append(Paragraph(f"Market Analysis Report for {symbol}", title_style))
        elements.append(Paragraph(f"Generated on {now}", normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Type of analysis
        elements.append(Paragraph(f"Analysis Type: {analysis_type} Technical Analysis", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Get data from the summary if available
        if summary:
            elements.append(Paragraph("Market Summary", heading_style))
            
            summary_data = [
                ["Metric", "Value"],
                ["Current Price", f"${summary['current_price']:.2f}"],
                ["Trend", summary['trend']],
                ["RSI", f"{summary['rsi']:.2f}" if summary['rsi'] is not None else "N/A"],
                ["RSI Condition", summary['rsi_condition']],
                ["MACD Signal", summary['macd_signal']],
                ["1-Day Change", f"{summary['price_1d_change']:.2f}%"],
                ["1-Week Change", f"{summary['price_1w_change']:.2f}%"],
                ["1-Month Change", f"{summary['price_1m_change']:.2f}%"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(summary_table)
            elements.append(Spacer(1, 0.25*inch))
            
        # If we have a prediction (from enhanced analysis)
        if prediction:
            elements.append(Paragraph("Price Prediction", heading_style))
            
            direction_text = "UP ↑" if prediction['direction'] > 0 else "DOWN ↓" if prediction['direction'] < 0 else "NEUTRAL ⟷"
            confidence = prediction['confidence'] * 100
            
            prediction_data = [
                ["Prediction Metric", "Value"],
                ["Predicted Direction", direction_text],
                ["Confidence", f"{confidence:.2f}%"],
                ["Current Price", f"${prediction['current_price']:.2f}"]
            ]
            
            if prediction['price_target']:
                price_change = ((prediction['price_target'] / prediction['current_price']) - 1) * 100
                prediction_data.append(["Price Target (5 days)", f"${prediction['price_target']:.2f} ({price_change:.2f}%)"])
            
            pred_table = Table(prediction_data, colWidths=[2*inch, 2*inch])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(pred_table)
            elements.append(Spacer(1, 0.25*inch))
            
        # Add data table if available
        if data is not None:
            elements.append(Paragraph("Recent Price Data", heading_style))
            
            # Get the last 5 days of data
            recent_data = data.iloc[-5:].reset_index()
            
            # Format DataFrame for table
            table_data = [["Date", "Open", "High", "Low", "Close", "Volume"]]
            for i, row in recent_data.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d')
                row_data = [
                    date_str,
                    f"${row['Open']:.2f}",
                    f"${row['High']:.2f}",
                    f"${row['Low']:.2f}",
                    f"${row['Close']:.2f}",
                    f"{row['Volume']:,.0f}"
                ]
                table_data.append(row_data)
            
            # Create table
            price_table = Table(table_data)
            price_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(price_table)
            elements.append(Spacer(1, 0.25*inch))
            
        # Add disclaimer
        elements.append(Spacer(1, 0.5*inch))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=normal_style,
            fontSize=8,
            textColor=colors.gray
        )
        disclaimer_text = """Disclaimer: This analysis is for informational purposes only and does not constitute investment advice. 
        Always do your own research before making investment decisions. Past performance is not indicative of future results."""
        elements.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Add footer
        elements.append(Spacer(1, 0.25*inch))
        footer_text = "EWTai Market Analyzer © 2025"
        elements.append(Paragraph(footer_text, disclaimer_style))
        
        # Build the PDF
        doc.build(elements)
        
        # Return the path to the PDF file
        return pdf_path
        
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

# Test function
if __name__ == "__main__":
    # Simple self-test
    print("Testing PDF generation...")
    
    # Create a mock data structure
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create mock data
    dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 110, 10),
        'High': np.random.uniform(105, 115, 10),
        'Low': np.random.uniform(95, 105, 10),
        'Close': np.random.uniform(100, 110, 10),
        'Volume': np.random.randint(1000000, 5000000, 10)
    })
    data.set_index('Date', inplace=True)
    
    # Create mock summary
    summary = {
        'current_price': 105.25,
        'trend': 'Bullish',
        'rsi': 65.5,
        'rsi_condition': 'Neutral',
        'macd_signal': 'Bullish',
        'price_1d_change': 1.25,
        'price_1w_change': 3.75,
        'price_1m_change': 8.5
    }
    
    # Create a test PDF
    pdf_path = create_simple_pdf('AAPL', data, summary=summary)
    if pdf_path:
        print(f"Test PDF created successfully at: {pdf_path}")
    else:
        print("Failed to create test PDF")