"""
Ultra simplified PDF generator that creates a pre-built PDF and returns its data directly.
This avoids all file system operations except for the actual PDF creation in memory.
"""

import io
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import pandas as pd

def generate_pdf_bytes(symbol, data=None, summary=None, prediction=None, wave_info=None, analysis_type="Technical Analysis"):
    """Generate a PDF and return the bytes directly - no file saving.
    
    Parameters:
    symbol (str): The stock symbol
    data (DataFrame): Price and indicator data
    summary (dict): Basic analysis summary
    prediction (dict): Price prediction data
    wave_info (dict): Elliott Wave analysis information
    """
    
    # Create a BytesIO buffer to store the PDF
    buffer = io.BytesIO()
    
    # Temp files to clean up
    temp_files = []
    
    # Handle any unexpected errors
    try:
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Create a function to safely convert numeric values
        def safe_float(value, default=0.0):
            try:
                return float(value)
            except:
                return default
                
        # Function to create a simple price chart and return the path to the image file
        def create_chart_image(symbol, data):
            try:
                # Create figure
                plt.figure(figsize=(7, 3.5))
                
                # Extract data
                dates = data.index
                prices = [safe_float(p) for p in data['Close']]
                
                # Plot price line
                plt.plot(dates, prices, 'b-', linewidth=2, label='Close Price')
                
                # Add moving averages if available
                if 'SMA_20' in data.columns:
                    sma20 = [safe_float(p) for p in data['SMA_20']]
                    plt.plot(dates, sma20, 'r-', linewidth=1, label='20-day SMA')
                    
                if 'SMA_50' in data.columns:
                    sma50 = [safe_float(p) for p in data['SMA_50']]
                    plt.plot(dates, sma50, 'g-', linewidth=1, label='50-day SMA')
                
                # Add title and labels
                plt.title(f"{symbol} Price Chart", fontsize=12)
                plt.xlabel("Date", fontsize=8)
                plt.ylabel("Price ($)", fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                
                # Format dates on x-axis
                plt.xticks(fontsize=7)
                plt.yticks(fontsize=8)
                plt.tight_layout()
                
                # Save to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_files.append(temp_file.name)
                plt.savefig(temp_file.name, dpi=150, format='png')
                plt.close()
                
                return temp_file.name
            except Exception as e:
                print(f"Error creating chart: {e}")
                return None
                
        def create_rsi_chart(data):
            try:
                # Check if RSI data is available
                if 'RSI' not in data.columns:
                    return None
                    
                # Create figure
                plt.figure(figsize=(7, 2))
                
                # Extract data
                dates = data.index
                rsi_values = [safe_float(r) for r in data['RSI']]
                
                # Plot RSI line
                plt.plot(dates, rsi_values, 'purple', linewidth=1.5, label='RSI')
                
                # Add overbought/oversold lines
                plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
                plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
                
                # Fill the middle area
                plt.fill_between(dates, 70, 30, color='gray', alpha=0.1)
                
                # Set axis limits and labels
                plt.ylim(0, 100)
                plt.title('RSI (Relative Strength Index)', fontsize=10)
                plt.ylabel('RSI Value', fontsize=8)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='upper right', fontsize=7)
                
                # Format dates
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=7)
                plt.tight_layout()
                
                # Save to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_files.append(temp_file.name)
                plt.savefig(temp_file.name, dpi=150, format='png')
                plt.close()
                
                return temp_file.name
            except Exception as e:
                print(f"Error creating RSI chart: {e}")
                return None
                
        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Custom styles for bullish/bearish signals
        bullish_style = normal_style.clone('Bullish')
        bullish_style.textColor = colors.green
        
        bearish_style = normal_style.clone('Bearish')
        bearish_style.textColor = colors.red
        
        # Title
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        elements.append(Paragraph(f"Market Analysis for {symbol}", title_style))
        elements.append(Paragraph(f"Generated on {now}", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Add analysis type
        elements.append(Paragraph(f"Analysis Type: {analysis_type}", heading_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # --- PRICE SUMMARY SECTION ---
        elements.append(Paragraph("Price Summary", heading_style))
        
        if data is not None:
            try:
                # Get basic price data
                current_price = safe_float(data['Close'].iloc[-1])
                
                # Calculate price changes if possible
                price_changes = []
                
                if len(data) > 1:
                    prev_close = safe_float(data['Close'].iloc[-2])
                    day_change = ((current_price / prev_close) - 1) * 100
                    price_changes.append(["1-Day Change", f"{day_change:.2f}%"])
                
                if len(data) > 5:
                    week_ago = safe_float(data['Close'].iloc[-5])
                    week_change = ((current_price / week_ago) - 1) * 100
                    price_changes.append(["1-Week Change", f"{week_change:.2f}%"])
                    
                if len(data) > 20:
                    month_ago = safe_float(data['Close'].iloc[-20])
                    month_change = ((current_price / month_ago) - 1) * 100
                    price_changes.append(["1-Month Change", f"{month_change:.2f}%"])
                
                # Create a simple table with price info
                price_data = [
                    ["Current Price", f"${current_price:.2f}"]
                ]
                
                # Add moving averages if available
                if 'SMA_20' in data.columns and not pd.isna(data['SMA_20'].iloc[-1]):
                    sma20 = safe_float(data['SMA_20'].iloc[-1])
                    relation = "Above MA (Bullish)" if current_price > sma20 else "Below MA (Bearish)"
                    price_data.append(["20-Day MA", f"${sma20:.2f} - {relation}"])
                    
                if 'SMA_50' in data.columns and not pd.isna(data['SMA_50'].iloc[-1]):
                    sma50 = safe_float(data['SMA_50'].iloc[-1])
                    price_data.append(["50-Day MA", f"${sma50:.2f}"])
                
                # Add price changes
                price_data.extend(price_changes)
                
                # Add high/low
                try:
                    high_price = safe_float(data['High'].max())
                    low_price = safe_float(data['Low'].min()) 
                    price_data.append(["Period Range", f"${low_price:.2f} - ${high_price:.2f}"])
                except:
                    pass
                    
                # Create and style the table
                price_table = Table(price_data, colWidths=[2*inch, 3*inch])
                price_table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                
                elements.append(price_table)
                elements.append(Spacer(1, 0.25*inch))
                
                # Add price chart
                elements.append(Paragraph("Price Chart", heading_style))
                
                # Generate and add price chart
                price_chart_path = create_chart_image(symbol, data)
                if price_chart_path:
                    img = Image(price_chart_path, width=6.5*inch, height=3.25*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.25*inch))
                else:
                    elements.append(Paragraph("Unable to generate price chart.", normal_style))
                    elements.append(Spacer(1, 0.25*inch))
                
            except Exception as e:
                elements.append(Paragraph(f"Error processing price data: {str(e)}", normal_style))
        
        # --- TECHNICAL INDICATORS SECTION ---
        if data is not None:
            elements.append(Paragraph("Technical Indicators", heading_style))
            
            try:
                indicators = []
                
                # Add RSI if available
                if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
                    rsi_val = safe_float(data['RSI'].iloc[-1])
                    
                    if rsi_val > 70:
                        rsi_condition = "Overbought (Bearish)"
                    elif rsi_val < 30:
                        rsi_condition = "Oversold (Bullish)"
                    else:
                        rsi_condition = "Neutral"
                        
                    indicators.append(["RSI", f"{rsi_val:.2f}", rsi_condition])
                
                # Add MACD if available
                if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
                    macd = safe_float(data['MACD'].iloc[-1])
                    macd_signal = safe_float(data['MACD_Signal'].iloc[-1])
                    
                    if macd > macd_signal:
                        macd_condition = "Bullish"
                    else:
                        macd_condition = "Bearish"
                        
                    indicators.append(["MACD", f"{macd:.4f}", macd_condition])
                    
                # Add stochastic if available
                if all(col in data.columns for col in ['%K', '%D']):
                    k_val = safe_float(data['%K'].iloc[-1])
                    d_val = safe_float(data['%D'].iloc[-1])
                    
                    if k_val < 20 and d_val < 20:
                        stoch_condition = "Oversold (Bullish)"
                    elif k_val > 80 and d_val > 80:
                        stoch_condition = "Overbought (Bearish)"
                    else:
                        stoch_condition = "Neutral"
                        
                    indicators.append(["Stochastic", f"%K: {k_val:.2f}, %D: {d_val:.2f}", stoch_condition])
                
                # If we have indicators, create a table
                if indicators:
                    # Add header
                    table_data = [["Indicator", "Value", "Interpretation"]]
                    table_data.extend(indicators)
                    
                    # Create and style the table
                    indicators_table = Table(table_data)
                    indicators_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ]))
                    
                    elements.append(indicators_table)
                    elements.append(Spacer(1, 0.25*inch))
                    
                    # Add RSI chart if RSI is available
                    if 'RSI' in data.columns:
                        elements.append(Paragraph("RSI Chart", heading_style))
                        
                        # Generate and add RSI chart
                        rsi_chart_path = create_rsi_chart(data)
                        if rsi_chart_path:
                            img = Image(rsi_chart_path, width=6.5*inch, height=1.9*inch)
                            elements.append(img)
                            elements.append(Spacer(1, 0.25*inch))
                        else:
                            elements.append(Paragraph("Unable to generate RSI chart.", normal_style))
                            elements.append(Spacer(1, 0.25*inch))
                else:
                    elements.append(Paragraph("Technical indicators not available", normal_style))
                    elements.append(Spacer(1, 0.1*inch))
            except Exception as e:
                elements.append(Paragraph(f"Error processing technical indicators: {str(e)}", normal_style))
        
        # --- PREDICTION SECTION (if available) ---
        if prediction:
            elements.append(Paragraph("Price Prediction", heading_style))
            
            try:
                direction = prediction.get('direction', 0)
                confidence = prediction.get('confidence', 0) * 100
                price_target = prediction.get('price_target')
                current = prediction.get('current_price', current_price if 'current_price' in locals() else 0)
                
                direction_text = "UP ↑" if direction > 0 else "DOWN ↓" if direction < 0 else "NEUTRAL ⟷"
                style = bullish_style if direction > 0 else (bearish_style if direction < 0 else normal_style)
                
                # Create prediction info
                elements.append(Paragraph(f"Predicted Direction: {direction_text}", style))
                elements.append(Paragraph(f"Confidence: {confidence:.1f}%", normal_style))
                
                if price_target:
                    change = ((price_target / current) - 1) * 100
                    elements.append(Paragraph(f"Price Target (5 days): ${price_target:.2f} ({change:.1f}%)", normal_style))
                
                # Add details if available
                if 'model_details' in prediction:
                    elements.append(Spacer(1, 0.1*inch))
                    elements.append(Paragraph("Prediction Factors:", normal_style))
                    
                    if 'technical_indicators' in prediction['model_details']:
                        tech = prediction['model_details']['technical_indicators']
                        
                        # Bullish signals
                        if 'bullish_signals' in tech and tech['bullish_signals']:
                            elements.append(Paragraph("Bullish Signals:", bullish_style))
                            for signal in tech['bullish_signals']:
                                elements.append(Paragraph(f"• {signal}", bullish_style))
                                
                        # Bearish signals
                        if 'bearish_signals' in tech and tech['bearish_signals']:
                            elements.append(Paragraph("Bearish Signals:", bearish_style))
                            for signal in tech['bearish_signals']:
                                elements.append(Paragraph(f"• {signal}", bearish_style))
                
                elements.append(Spacer(1, 0.25*inch))
            except Exception as e:
                elements.append(Paragraph(f"Error processing prediction: {str(e)}", normal_style))
                
        # --- ELLIOTT WAVE ANALYSIS SECTION (if available) ---
        if wave_info and wave_info.get('patterns'):
            elements.append(Paragraph("Elliott Wave Analysis", heading_style))
            
            try:
                patterns = wave_info['patterns']
                latest_pattern = patterns[-1]  # Use the most recent pattern
                
                # Determine if bullish or bearish
                is_bullish = latest_pattern['wave_points'][5] > latest_pattern['wave_points'][0]
                direction_text = "bullish" if is_bullish else "bearish"
                
                # Get pattern details
                pattern_type = latest_pattern.get('type', 'impulse+correction')
                confidence = latest_pattern.get('confidence', 0.7)
                
                # Get key price levels
                wave_points = latest_pattern['wave_points']
                wave1_price = wave_points[0]
                wave3_price = wave_points[3]
                wave5_price = wave_points[5]
                
                # Calculate Fibonacci extensions for targets
                wave1_2_range = abs(wave_points[2] - wave_points[0])
                fib_1382_target = wave1_price + (wave1_2_range * 1.382) if is_bullish else wave1_price - (wave1_2_range * 1.382)
                                            
                wave3_4_range = abs(wave_points[4] - wave_points[3])
                fib_1236_target = wave3_price + (wave3_4_range * 1.236) if is_bullish else wave3_price - (wave3_4_range * 1.236)
                fib_1386_target = wave3_price + (wave3_4_range * 1.386) if is_bullish else wave3_price - (wave3_4_range * 1.386)
                
                # Create detailed wave analysis
                wave_text = f"""
                Current Price Action Analysis
                
                I want to explain how important the region between ${min(fib_1236_target, fib_1386_target):.2f} and ${max(fib_1236_target, fib_1386_target):.2f} is in this {symbol}'s price history. 
                Elliott Wave analysts look for price structures to culminate at the confluence of Fibonacci levels. These levels are derived from targets of multiple degrees.
                Confluence across degrees gives us high probability zones for price to either top or bottom.
                
                I see multiple degrees of confluence in this region:
                
                - The 1.382 extension of waves one and two is at ${fib_1382_target:.2f}. This is a key target zone.
                
                - After completing a third and fourth wave, prices tend to top/bottom between the 1.236 to 1.386 extension of wave three.
                  This gives us a target zone of ${fib_1236_target:.2f} to ${fib_1386_target:.2f}.
                
                Currently, we are watching the development of wave 5. The structure suggests the price is in a {direction_text} trend overall.
                """
                
                elements.append(Paragraph(wave_text, normal_style))
                elements.append(Spacer(1, 0.2*inch))
                
                # Add key price levels
                if 'support_resistance' in wave_info:
                    sr_levels = wave_info['support_resistance']
                    
                    # Add a table of key price levels
                    level_data = [["Level Type", "Price", "Description"]]
                    
                    # Add resistance levels
                    if 'key_resistance' in sr_levels and sr_levels['key_resistance']:
                        for level in sr_levels['key_resistance']:
                            level_data.append(["Resistance", f"${level['price']:.2f}", level['description']])
                    
                    # Add support levels
                    if 'key_support' in sr_levels and sr_levels['key_support']:
                        for level in sr_levels['key_support']:
                            level_data.append(["Support", f"${level['price']:.2f}", level['description']])
                            
                    # Add invalidation levels
                    if 'invalidation_levels' in sr_levels and sr_levels['invalidation_levels']:
                        for level in sr_levels['invalidation_levels']:
                            level_data.append(["Invalidation", f"${level['price']:.2f}", level['description']])
                            
                    # Create and style the table
                    if len(level_data) > 1:  # Only if we have data beyond the header
                        elements.append(Paragraph("Key Price Levels", heading_style))
                        levels_table = Table(level_data)
                        levels_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ]))
                        elements.append(levels_table)
                        elements.append(Spacer(1, 0.2*inch))
                
                # Add stop-loss recommendations
                if 'support_resistance' in wave_info and 'stop_loss_levels' in wave_info['support_resistance']:
                    stop_levels = wave_info['support_resistance']['stop_loss_levels']
                    
                    elements.append(Paragraph("Strategic Stop-Loss Levels", heading_style))
                    
                    stop_data = [["Entry Point", "Stop-Loss Price", "Description"]]
                    for level in stop_levels:
                        stop_data.append([level['for_entry'], f"${level['price']:.2f}", level['description']])
                    
                    stop_table = Table(stop_data)
                    stop_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('BACKGROUND', (0, 1), (0, -1), colors.lightpink),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ]))
                    elements.append(stop_table)
                
                elements.append(Spacer(1, 0.25*inch))
                
                # Add retracement levels
                if 'retracements' in wave_info and wave_info['retracements']:
                    retracements = wave_info['retracements'][-1]  # Use the most recent
                    
                    elements.append(Paragraph("Fibonacci Retracement Levels", heading_style))
                    
                    fib_text = f"Impulse Move: ${retracements['impulse_range'][0]:.2f} to ${retracements['impulse_range'][1]:.2f}"
                    elements.append(Paragraph(fib_text, normal_style))
                    
                    # Add retracement levels in a table
                    fib_data = [["Fibonacci Level", "Price"]]
                    for fib, level in retracements['retracement_levels'].items():
                        fib_data.append([f"{fib*100:.1f}%", f"${level:.2f}"])
                    
                    fib_table = Table(fib_data)
                    fib_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ]))
                    elements.append(fib_table)
                
                elements.append(Spacer(1, 0.25*inch))
                
            except Exception as e:
                elements.append(Paragraph(f"Error processing Elliott Wave analysis: {str(e)}", normal_style))
        
        # --- RECENT DATA SECTION ---
        elements.append(Paragraph("Recent Price Data", heading_style))
        
        if data is not None:
            try:
                # Alternative approach for the price data table
                # Instead of using reset_index, we'll create the table manually
                
                # Get the last 5 rows of data
                last_5_days = data.tail(5)
                
                # Create header row
                table_data = [["Date", "Open", "High", "Low", "Close", "Volume"]]
                
                # Get the index values (dates) for display
                try:
                    index_dates = last_5_days.index.tolist()
                except:
                    # Fallback if we can't access the index
                    index_dates = ["N/A"] * len(last_5_days)
                
                # Manually iterate through the rows of data
                for i in range(len(last_5_days)):
                    try:
                        # Get the date from the index
                        date_value = index_dates[i] if i < len(index_dates) else "N/A"
                        
                        # Format the date
                        if hasattr(date_value, 'strftime'):
                            date_str = date_value.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date_value).split()[0]  # Try to get just the date portion
                        
                        # Get row values safely
                        try:
                            open_val = safe_float(last_5_days['Open'].iloc[i])
                            high_val = safe_float(last_5_days['High'].iloc[i])
                            low_val = safe_float(last_5_days['Low'].iloc[i])
                            close_val = safe_float(last_5_days['Close'].iloc[i])
                            volume_val = safe_float(last_5_days['Volume'].iloc[i])
                        except:
                            # If we can't get specific columns, try dict-like access
                            open_val = safe_float(last_5_days.iloc[i].get('Open', 0))
                            high_val = safe_float(last_5_days.iloc[i].get('High', 0))
                            low_val = safe_float(last_5_days.iloc[i].get('Low', 0))
                            close_val = safe_float(last_5_days.iloc[i].get('Close', 0))
                            volume_val = safe_float(last_5_days.iloc[i].get('Volume', 0))
                        
                        # Add the formatted row to the table
                        table_data.append([
                            date_str,
                            f"${open_val:.2f}",
                            f"${high_val:.2f}",
                            f"${low_val:.2f}",
                            f"${close_val:.2f}",
                            f"{int(volume_val):,}"
                        ])
                    except Exception as row_error:
                        # Add a placeholder row if we hit any errors
                        table_data.append(["Error", "$0.00", "$0.00", "$0.00", "$0.00", "0"])
                        elements.append(Paragraph(f"Error processing row: {str(row_error)}", normal_style))
                
                # Create and style the table
                recent_table = Table(table_data)
                recent_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                
                elements.append(recent_table)
            except Exception as e:
                elements.append(Paragraph(f"Error creating data table: {str(e)}", normal_style))
        
        # --- ICHIMOKU CLOUD ANALYSIS ---
        elements.append(Paragraph("Ichimoku Cloud Analysis", heading_style))
        
        # Create a table for Ichimoku signals
        ichimoku_signals = [
            ["Condition", "Signal"],
            ["Price above cloud", "Strong Bullish"],
            ["Price below cloud", "Strong Bearish"],
            ["Price inside cloud", "Neutral/Transitioning"],
            ["Tenkan-sen above Kijun-sen", "Bullish Cross (Buy Signal)"],
            ["Tenkan-sen below Kijun-sen", "Bearish Cross (Sell Signal)"],
            ["Cloud color green (Span A > Span B)", "Bullish Future Cloud"],
            ["Cloud color red (Span A < Span B)", "Bearish Future Cloud"]
        ]
        
        ichimoku_table = Table(ichimoku_signals, colWidths=[2.5*inch, 3*inch])
        ichimoku_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            # Color the bullish and bearish signals
            ('BACKGROUND', (1, 1), (1, 1), colors.lightgreen),
            ('BACKGROUND', (1, 2), (1, 2), colors.lightcoral),
            ('BACKGROUND', (1, 4), (1, 4), colors.lightgreen),
            ('BACKGROUND', (1, 5), (1, 5), colors.lightcoral),
            ('BACKGROUND', (1, 6), (1, 6), colors.lightgreen),
            ('BACKGROUND', (1, 7), (1, 7), colors.lightcoral)
        ]))
        
        elements.append(ichimoku_table)
        elements.append(Spacer(1, 0.25*inch))
        
        # --- COMBINED STRATEGY SECTION ---
        elements.append(Paragraph("Combined Elliott Wave and Ichimoku Strategy", heading_style))
        elements.append(Paragraph("Best Trading Scenarios:", normal_style))
        elements.append(Paragraph("• Wave 3 confirmed + Price above cloud + Bullish TK Cross = Strong Buy", bullish_style))
        elements.append(Paragraph("• Early Wave 5 + Price above cloud = Continue holding", bullish_style))
        elements.append(Paragraph("• Late Wave 5 + Price entering cloud = Prepare to exit", normal_style))
        elements.append(Paragraph("• Wave A confirmed + Price below cloud + Bearish TK Cross = Strong Sell", bearish_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # --- CABPIPO SECTION ---
        elements.append(Paragraph("IPO Consideration", heading_style))
        elements.append(Paragraph("Elliott Wave patterns may have begun before a company's IPO. For recently public companies, the CABpIPO pattern (Cow A. Bunga post IPO pattern) explains common wave structures that emerge after IPOs. This often involves a significant corrective phase (A-B-C) after the initial public offering, followed by a new impulse pattern.", normal_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # --- DISCLAIMER AND FOOTER ---
        elements.append(Spacer(1, 0.5*inch))
        disclaimer_style = normal_style.clone('Disclaimer')
        disclaimer_style.fontSize = 8
        disclaimer_style.textColor = colors.gray
        
        disclaimer = "Disclaimer: This analysis is for informational purposes only and does not constitute investment advice. Always do your own research before making investment decisions. Past performance is not indicative of future results."
        elements.append(Paragraph(disclaimer, disclaimer_style))
        
        elements.append(Spacer(1, 0.25*inch))
        elements.append(Paragraph("EWTai Market Analyzer © 2025", disclaimer_style))
        
        # Build the PDF
        doc.build(elements)
        
        # Get the PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
        
    except Exception as e:
        # If anything goes wrong, create a super-simple error PDF
        try:
            # Close the previous buffer if it exists
            buffer.close()
            
            # Create a new buffer
            error_buffer = io.BytesIO()
            
            # Create a simple PDF with just error information
            simple_doc = SimpleDocTemplate(error_buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            
            error_elements = []
            error_elements.append(Paragraph(f"Market Analysis for {symbol}", styles['Title']))
            error_elements.append(Spacer(1, 0.5*inch))
            error_elements.append(Paragraph("Error Creating Complete Report", styles['Heading2']))
            error_elements.append(Paragraph(f"There was an error processing some data: {str(e)}", styles['Normal']))
            error_elements.append(Spacer(1, 0.5*inch))
            error_elements.append(Paragraph("The analysis is still available in the interactive web app.", styles['Normal']))
            
            # Build the simple PDF
            simple_doc.build(error_elements)
            
            # Get the PDF bytes
            error_pdf_bytes = error_buffer.getvalue()
            error_buffer.close()
            
            return error_pdf_bytes
            
        except:
            # If even the error PDF fails, return an empty PDF
            from reportlab.pdfgen import canvas
            last_buffer = io.BytesIO()
            c = canvas.Canvas(last_buffer)
            c.drawString(100, 750, f"Error generating report for {symbol}")
            c.save()
            minimal_pdf = last_buffer.getvalue()
            last_buffer.close()
            return minimal_pdf

# Test function
if __name__ == "__main__":
    # Create test data
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create mock data
    dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
    mock_data = pd.DataFrame({
        'Date': dates,
        'Close': np.random.uniform(100, 110, 10),
        'Open': np.random.uniform(100, 110, 10),
        'High': np.random.uniform(105, 115, 10),
        'Low': np.random.uniform(95, 105, 10),
        'Volume': np.random.randint(1000000, 5000000, 10)
    })
    mock_data.set_index('Date', inplace=True)
    
    # Add some technical indicators
    mock_data['RSI'] = np.random.uniform(40, 60, 10)
    mock_data['SMA_20'] = np.random.uniform(100, 110, 10)
    
    # Generate a test PDF
    pdf_bytes = generate_pdf_bytes('AAPL', mock_data)
    
    # Write to a test file to verify
    with open('test_direct.pdf', 'wb') as f:
        f.write(pdf_bytes)
    
    print(f"Test PDF created with {len(pdf_bytes)} bytes")
    print("Saved to test_direct.pdf for verification")