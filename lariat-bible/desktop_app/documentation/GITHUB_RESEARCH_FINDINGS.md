# GitHub Research Findings for The Lariat Bible

> **Comprehensive survey of GitHub repositories and tools to enhance our restaurant management system**
> Research Date: 2025-11-18

## Executive Summary

This document compiles findings from GitHub public repositories that can help develop and improve The Lariat Bible's algorithms, processes, and features. The research focused on:

1. Restaurant management systems
2. Food distributor integrations (SYSCO/Shamrock)
3. Recipe costing and food cost calculators
4. Excel/CSV parsers
5. Invoice OCR and data extraction
6. Vendor comparison tools
7. Fuzzy matching algorithms
8. Financial reporting and dashboards
9. ETL pipelines and data validation
10. Maintenance tracking systems
11. Time series forecasting
12. Unit conversion libraries

---

## 1. Restaurant Management Systems

### Key Findings

**No SYSCO or Shamrock-specific tools found** - There are no public GitHub repositories specifically for parsing SYSCO or Shamrock Foods order guides. This means we'll need to build custom parsers.

### Notable Projects

#### 1. **Restaurant-Management-Kivy** by Ahmed-Fawzy
- **Repository**: https://github.com/Ahmed-Fawzy/Restaurant-Management-Kivy
- **Tech Stack**: Python, Kivy
- **Features**: Open-source restaurant management system
- **Relevance**: General structure and patterns for restaurant operations

#### 2. **foody** by pythonbrad
- **Repository**: https://github.com/pythonbrad/foody
- **Tech Stack**: Python
- **Features**: Restaurant management system
- **Relevance**: Architecture patterns

#### 3. **RestaurantManagementSystem** by nurlan-aliyev
- **Repository**: https://github.com/nurlan-aliyev/RestaurantManagementSystem
- **Tech Stack**: Python, Tkinter, SQLite3
- **Features**: Order tracking, database storage for analysis
- **Relevance**: Database design patterns for orders and inventory

### Recommendations for The Lariat Bible

✅ **Implement**:
- Study database schemas from these projects
- Adopt SQLite patterns for development, PostgreSQL for production
- Review order tracking and data analysis approaches

❌ **Avoid**:
- GUI frameworks like Tkinter (we're using Flask web interface)
- Monolithic architecture (we have modular design)

---

## 2. Recipe Costing & Food Cost Calculators

### Top GitHub Repositories

#### 1. **food-calculator-api** by petritz
- **Repository**: https://github.com/petritz/food-calculator-api
- **Features**: Calculate price, calories, etc. of recipes
- **Best for**: Pastries and baked goods
- **What to learn**: API structure for recipe calculations

#### 2. **Recipe-Cost-Calculator** by gmoraitis
- **Repository**: https://github.com/gmoraitis/Recipe-Cost-Calculator
- **Features**: Add prices and grams, see cost per ingredient and total
- **What to learn**: Simple ingredient costing algorithms

#### 3. **cost-calculator** by jelaniwoods
- **Repository**: https://github.com/jelaniwoods/cost-calculator
- **Features**: Recipe & shopping list cost calculator
- **What to learn**: Shopping list integration

#### 4. **recipe_costs_frontend** by aparkening
- **Repository**: https://github.com/aparkening/recipe_costs_frontend
- **Features**: JavaScript SPA for restaurant owners to auto-price recipes
- **What to learn**: Frontend patterns for recipe pricing

### Recommendations for The Lariat Bible

✅ **Implement**:
- Review algorithms for per-portion cost calculation
- Study how they handle ingredient scaling
- Examine margin calculation approaches

📝 **Code Examples to Study**:
```python
# From Stack Overflow discussion on recipe costing
# https://stackoverflow.com/questions/68263107/get-cost-of-ingredients-in-a-recipe

# Pattern we should adopt:
class RecipeIngredient:
    def __init__(self, ingredient, quantity, unit):
        self.ingredient = ingredient
        self.quantity = quantity
        self.unit = unit

    def cost(self):
        # Convert to base unit and calculate cost
        return self.ingredient.unit_price * self.quantity_in_base_unit()
```

---

## 3. Invoice OCR & Data Extraction

### Top GitHub Repositories

#### 1. **invoice-parser** by aj-jhaveri ⭐ **HIGHLY RECOMMENDED**
- **Repository**: https://github.com/aj-jhaveri/invoice-parser
- **Features**:
  - Processes both text-based and scanned PDF invoices
  - Uses pdf2image + pytesseract for OCR
  - Extracts invoice number, vendor, date, total amount
  - Saves to CSV
- **What to learn**: Complete invoice processing pipeline
- **Why relevant**: Exactly what we need for SYSCO/Shamrock invoices

#### 2. **Invoice_ocr** by anshumyname ⭐ **RECOMMENDED**
- **Repository**: https://github.com/anshumyname/Invoice_ocr
- **Features**:
  - Converts scanned invoices to Excel
  - Uses cv2 + pytesseract
  - Detects horizontal/vertical lines for tables
  - Constructs tabular data
- **What to learn**: Table detection and extraction from invoices
- **Why relevant**: Vendor invoices are often tabular

#### 3. **invoice2data** by invoice-x
- **Repository**: https://github.com/invoice-x/invoice2data
- **Features**:
  - Template-based invoice extraction
  - YAML/JSON configuration
  - Command line tool and Python library
- **What to learn**: Template-based approach for consistent vendors
- **Why relevant**: Once we have SYSCO/Shamrock templates, this is perfect

#### 4. **OCR-Invoice-Detection** by garrlicbread
- **Repository**: https://github.com/garrlicbread/OCR-Invoice-Detection
- **Features**: Simple script with OpenCV and PyTesseract
- **What to learn**: Image preprocessing techniques

### Recommendations for The Lariat Bible

✅ **Implement Immediately**:
1. Start with `invoice-parser` approach for quick wins
2. Add OpenCV preprocessing from `Invoice_ocr` for better accuracy
3. Build templates using `invoice2data` approach once we have samples

📝 **Implementation Plan**:
```python
# modules/vendor_analysis/invoice_processor.py

from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import pandas as pd

class InvoiceProcessor:
    def __init__(self):
        self.templates = {
            'SYSCO': self.load_sysco_template(),
            'Shamrock Foods': self.load_shamrock_template()
        }

    def process_invoice(self, pdf_path, vendor):
        # 1. Convert PDF to images
        images = convert_from_path(pdf_path)

        # 2. Preprocess with OpenCV
        preprocessed = self.preprocess_image(images[0])

        # 3. OCR with pytesseract
        text = pytesseract.image_to_string(preprocessed)

        # 4. Extract data using template
        data = self.extract_data(text, vendor)

        # 5. Return structured data
        return data
```

---

## 4. Excel/CSV Parsing & Python Automation

### Core Libraries

#### 1. **xlwings** ⭐ **TOP CHOICE**
- **Repository**: https://github.com/xlwings/xlwings
- **Features**:
  - Call Python from Excel and vice versa
  - Works on Windows, macOS, web, and Google Sheets
  - Full support for pandas DataFrames
  - Can automate Excel tasks and replace VBA macros
- **What to learn**: Bidirectional Excel-Python integration
- **Use case**: Reading vendor order guides, exporting comparison reports

#### 2. **openpyxl** (Already in requirements.txt ✅)
- **Features**:
  - Read/write Excel files
  - Advanced formatting, conditional formatting
  - Worksheet manipulation
- **What to learn**: Deep Excel file manipulation
- **Use case**: Creating formatted vendor comparison reports

#### 3. **pandas** (Already in requirements.txt ✅)
- **Features**:
  - Data manipulation and analysis
  - Read Excel, CSV easily
  - Powerful data processing
- **What to learn**: Data analysis workflows
- **Use case**: Vendor price analysis, trend detection

### Integration Pattern

```python
# Best practice: Use them together
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

# Read with pandas for data processing
df = pd.read_excel('sysco_order_guide.xlsx')

# Process data
df['savings'] = df['sysco_price'] - df['shamrock_price']
df['savings_pct'] = (df['savings'] / df['sysco_price']) * 100

# Write with openpyxl for formatting
with pd.ExcelWriter('vendor_comparison.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Comparison')

    # Get workbook and apply formatting
    workbook = writer.book
    worksheet = writer.sheets['Comparison']

    # Highlight savings over 20%
    for row in worksheet.iter_rows(min_row=2):
        if row[5].value > 20:  # savings_pct column
            row[5].fill = PatternFill(start_color="00FF00", fill_type="solid")
```

### Recommendations for The Lariat Bible

✅ **Already doing well**: We have pandas and openpyxl in requirements.txt

🔧 **Consider adding**:
- xlwings for interactive Excel workflows (optional)
- python-excel libraries for CSV edge cases

---

## 5. Vendor Price Comparison & Analysis

### Key Repositories

#### 1. **CompareGo** by imsudip
- **Repository**: https://github.com/imsudip/CompareGo
- **Features**:
  - Price comparison between e-commerce sites
  - Sentiment analysis on reviews
  - Web scraping with BeautifulSoup
- **What to learn**: Multi-vendor comparison algorithms

#### 2. **Price-comparison-website** by smartyrad
- **Repository**: https://github.com/smartyrad/Price-comparison-website-using-web-scraping-in-python
- **Features**: Real-time price comparison using web scraping
- **What to learn**: Automated price tracking

#### 3. **price-comparison-project** by RamonWill
- **Repository**: https://github.com/RamonWill/price-comparison-project
- **Features**: Django webscraper for UK supermarkets
- **What to learn**: Database updates with latest prices

### Recommendations for The Lariat Bible

✅ **Apply these concepts**:
1. Price trend tracking over time
2. Automated alerts when prices change significantly
3. Historical price database for analysis

📝 **Enhancement Idea**:
```python
# modules/vendor_analysis/price_tracker.py

class PriceTracker:
    def track_price_changes(self, product_id, vendor, new_price):
        """Track price changes and alert on significant changes"""
        old_price = self.get_latest_price(product_id, vendor)

        if old_price:
            change_pct = ((new_price - old_price) / old_price) * 100

            # Alert if price changed more than 10%
            if abs(change_pct) > 10:
                self.send_alert(product_id, vendor, old_price, new_price, change_pct)

        # Store in price history
        self.price_history.append({
            'product_id': product_id,
            'vendor': vendor,
            'price': new_price,
            'timestamp': datetime.now()
        })
```

---

## 6. Fuzzy Matching for Product Comparison ⭐ **CRITICAL**

### Libraries

#### 1. **RapidFuzz** ⭐ **RECOMMENDED**
- **Repository**: https://github.com/rapidfuzz/RapidFuzz
- **Features**:
  - Fast string matching (C++ backend)
  - Drop-in replacement for FuzzyWuzzy
  - MIT licensed
  - Better performance
- **Install**: `pip install rapidfuzz`
- **Why**: Faster than FuzzyWuzzy, actively maintained

#### 2. **TheFuzz** (FuzzyWuzzy renamed)
- **Repository**: https://github.com/seatgeek/thefuzz
- **Features**:
  - Original fuzzy matching library
  - Levenshtein Distance calculations
  - Pure Python
- **Install**: `pip install thefuzz`
- **Why**: Well-documented, widely used

### Comparison

| Feature | RapidFuzz | TheFuzz |
|---------|-----------|---------|
| Speed | ⚡⚡⚡⚡⚡ Very Fast | ⚡⚡⚡ Moderate |
| Accuracy | ✅ Same | ✅ Same |
| License | MIT | GPL-2.0 |
| Active | ✅ Yes | ⚠️ Less active |

### Why This Matters for The Lariat Bible

The product matching verification document shows we need to match products like:
- "Black Pepper Fine" vs "Black Pepper Fine" ✅
- "Ground Beef 80/20" vs "Beef Ground 80/20" ✅
- "Chicken Breast Boneless" vs "Chicken Breast BNLS" ✅

But NOT match:
- "Black Pepper Fine" vs "Black Pepper Coarse" ❌

### Implementation Example

```python
# Add to requirements.txt
# rapidfuzz==3.5.2

from rapidfuzz import fuzz, process

class ProductMatcher:
    """Match products between SYSCO and Shamrock"""

    def match_product(self, sysco_product, shamrock_products, threshold=85):
        """
        Find best matching Shamrock product for a SYSCO product

        Args:
            sysco_product: SYSCO product description
            shamrock_products: List of Shamrock product descriptions
            threshold: Minimum similarity score (0-100)

        Returns:
            Best match and score, or None if below threshold
        """
        # Use token_sort_ratio to handle word order differences
        # "Ground Beef 80/20" matches "Beef Ground 80/20"
        result = process.extractOne(
            sysco_product,
            shamrock_products,
            scorer=fuzz.token_sort_ratio
        )

        if result and result[1] >= threshold:
            return {
                'match': result[0],
                'score': result[1],
                'confidence': 'high' if result[1] > 90 else 'medium'
            }

        return None

    def verify_match(self, sysco_product, shamrock_product):
        """
        Verify a specific match with multiple scoring methods
        """
        scores = {
            'exact': fuzz.ratio(sysco_product, shamrock_product),
            'partial': fuzz.partial_ratio(sysco_product, shamrock_product),
            'token_sort': fuzz.token_sort_ratio(sysco_product, shamrock_product),
            'token_set': fuzz.token_set_ratio(sysco_product, shamrock_product)
        }

        return {
            'scores': scores,
            'average': sum(scores.values()) / len(scores),
            'recommended': scores['token_sort'] > 85
        }
```

### Recommendations for The Lariat Bible

✅ **Critical Implementation**:
1. Add `rapidfuzz` to requirements.txt
2. Implement ProductMatcher class
3. Use for initial order guide matching
4. **Manual verification** for matches with score 70-85
5. Auto-accept matches > 90
6. Auto-reject matches < 70

⚠️ **Important**: As noted in PRODUCT_MATCHING_VERIFICATION.md:
- Different grinds are DIFFERENT products
- Quality differences matter
- Use fuzzy matching to find CANDIDATES, then manual review

---

## 7. Inventory Management for Restaurants

### Key Repository

#### **Restaurant-Ingredient-Tracker** by shaecodes ⭐ **HIGHLY RELEVANT**
- **Repository**: https://github.com/shaecodes/Restaurant-Ingredient-Tracker
- **Features**:
  - Ingredient inventory management for restaurants
  - Restocking alerts
  - Stock level monitoring
  - Usage tracking
  - Log tracking
- **What to learn**: Restaurant-specific inventory patterns

#### **InvenTree**
- **Repository**: https://github.com/inventree/InvenTree
- **Features**:
  - Open-source inventory management
  - Python/Django backend
  - REST API
  - Low-level stock control
  - Part tracking
- **What to learn**: Advanced inventory system architecture

### Recommendations for The Lariat Bible

✅ **Features to implement**:
```python
# modules/inventory/stock_manager.py

class StockManager:
    def check_reorder_point(self, ingredient_id):
        """Alert when ingredient hits reorder point"""
        ingredient = self.get_ingredient(ingredient_id)
        current_stock = ingredient.current_quantity
        reorder_point = ingredient.reorder_point
        lead_time_days = ingredient.lead_time_days

        if current_stock <= reorder_point:
            # Calculate order quantity
            daily_usage = self.calculate_average_daily_usage(ingredient_id)
            order_qty = (daily_usage * lead_time_days) * 1.5  # 1.5x safety factor

            return {
                'alert': True,
                'current_stock': current_stock,
                'suggested_order': order_qty,
                'vendor': ingredient.preferred_vendor,
                'estimated_cost': order_qty * ingredient.best_unit_price
            }

        return {'alert': False}

    def track_usage(self, ingredient_id, quantity_used, recipe_id=None):
        """Track ingredient usage for analysis"""
        self.usage_log.append({
            'ingredient_id': ingredient_id,
            'quantity': quantity_used,
            'recipe_id': recipe_id,
            'timestamp': datetime.now(),
            'remaining_stock': self.get_current_stock(ingredient_id)
        })
```

---

## 8. Data Visualization & Business Dashboards

### Streamlit + Plotly ⭐ **RECOMMENDED STACK**

#### Key Resources

1. **Admin-dashboard** by korenkaplan
   - **Repository**: https://github.com/korenkaplan/Admin-dashboard
   - **Features**: Dynamic interactive dashboard with Streamlit + Plotly
   - **What to learn**: Dashboard layout patterns

2. **personal-finance-dashboard** by vinzalfaro
   - **Repository**: https://github.com/vinzalfaro/personal-finance-dashboard
   - **Features**: Streamlit + Plotly + SQL for financial data
   - **What to learn**: Financial KPI visualization

### Why Streamlit + Plotly?

**Pros**:
- 🚀 Fast development (dashboard in < 50 lines)
- 📊 Beautiful interactive charts
- 🔄 Real-time updates
- 🌐 Easy deployment (Streamlit Cloud)
- 🐍 Pure Python (no JavaScript needed)

**Cons**:
- ⚠️ Different from Flask (would need separate app)
- ⚠️ Less customization than custom frontend

### Integration with The Lariat Bible

```python
# Option 1: Keep Flask API, add Streamlit dashboard
# Run Flask on :5000, Streamlit on :8501

# streamlit_dashboard.py
import streamlit as st
import plotly.express as px
import requests

# Fetch data from Flask API
response = requests.get('http://localhost:5000/api/vendor-comparison')
data = response.json()

# Create dashboard
st.title("🤠 The Lariat Bible Dashboard")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Monthly Catering Revenue", "$28,000")
col2.metric("Monthly Savings", f"${data['monthly_savings']:,.2f}")
col3.metric("Annual Savings", f"${data['annual_savings']:,.2f}")

# Chart
fig = px.bar(df, x='category', y='savings', title='Savings by Category')
st.plotly_chart(fig)
```

### Recommendations for The Lariat Bible

🤔 **Decision Point**:
- **Keep Flask** for backend API (current choice ✅)
- **Add Streamlit** for advanced dashboards (optional, future)
- **Use Plotly** in Flask templates for charts (compromise)

---

## 9. ETL Pipelines & Data Validation

### Key Repositories

#### 1. **Building-ETL-Pipelines-with-Python** (Packt Publishing)
- **Repository**: https://github.com/PacktPublishing/Building-ETL-Pipelines-with-Python
- **Features**: Design patterns for robust ETL
- **What to learn**: Data cleansing, transformation techniques

#### 2. **Crowdfunding_ETL** by Hamim-Hussain
- **Repository**: https://github.com/Hamim-Hussain/Crowdfunding_ETL
- **Features**:
  - Extract from multiple sources
  - Clean with pandas, numpy, datetime
  - Load to PostgreSQL
- **What to learn**: Complete ETL workflow

### Pydantic for Data Validation ⭐ **CRITICAL**

#### Why Pydantic?

- ✅ Already in requirements.txt
- ✅ Used by 466,400+ GitHub repositories
- ✅ Used by NASA, Google, Microsoft, IBM
- ✅ Type hints for validation
- ✅ Fast (written in Rust)
- ✅ Excellent error messages

#### Implementation for The Lariat Bible

```python
# modules/vendor_analysis/schemas.py

from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional

class VendorProduct(BaseModel):
    """Schema for vendor product data validation"""

    product_code: str = Field(..., min_length=1, max_length=50)
    description: str = Field(..., min_length=1)
    vendor: str = Field(..., pattern=r'^(SYSCO|Shamrock Foods)$')
    pack_size: str
    case_price: float = Field(..., gt=0)
    unit_price: float = Field(..., gt=0)
    unit: str = Field(..., pattern=r'^(LB|OZ|EA|GAL|QT)$')
    category: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.now)

    @validator('case_price', 'unit_price')
    def validate_prices(cls, v):
        """Ensure prices are reasonable"""
        if v > 10000:
            raise ValueError('Price seems too high, please verify')
        return round(v, 2)

    @validator('unit_price', always=True)
    def validate_unit_price(cls, v, values):
        """Unit price should be less than case price"""
        if 'case_price' in values and v > values['case_price']:
            raise ValueError('Unit price cannot exceed case price')
        return v

class Recipe(BaseModel):
    """Schema for recipe data"""

    recipe_id: str
    name: str = Field(..., min_length=1, max_length=200)
    category: str
    yield_amount: float = Field(..., gt=0)
    yield_unit: str
    ingredients: list

    class Config:
        json_schema_extra = {
            "example": {
                "recipe_id": "REC001",
                "name": "BBQ Sauce",
                "category": "Sauce",
                "yield_amount": 1.0,
                "yield_unit": "gallon",
                "ingredients": []
            }
        }

# Usage
try:
    product = VendorProduct(
        product_code="SYS001",
        description="Ground Beef 80/20",
        vendor="SYSCO",
        pack_size="10 LB",
        case_price=45.99,
        unit_price=4.599,
        unit="LB",
        category="MEAT"
    )
except ValidationError as e:
    print(f"Data validation failed: {e}")
```

### Recommendations for The Lariat Bible

✅ **Implement immediately**:
1. Create Pydantic models for all data structures
2. Validate data at API boundaries
3. Use for invoice parsing output validation
4. Validate order guide imports

---

## 10. Maintenance Tracking & Scheduling

### Key Repositories

#### 1. **Machine-Predictive-Maintenance** by Adhiban1
- **Repository**: https://github.com/Adhiban1/Machine-Predictive-Maintenance
- **Features**:
  - Data analysis to predict failures
  - Preventive maintenance scheduling
  - Minimize downtime
- **What to learn**: Predictive maintenance algorithms

#### 2. **ML-Based Vehicle Predictive Maintenance** by iDharshan
- **Repository**: https://github.com/iDharshan/ML-Based-Vehicle-Predictive-Maintenance-System-with-Real-Time-Visualization
- **Features**:
  - GBM models for failure prediction
  - Streamlit web interface
  - Real-time sensor data
- **What to learn**: ML-based maintenance prediction

### Recommendations for The Lariat Bible

📝 **Simple implementation first, ML later**:

```python
# modules/equipment/maintenance_scheduler.py

from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class MaintenanceSchedule:
    equipment_id: str
    task: str
    frequency_days: int
    last_completed: datetime
    estimated_duration_hours: float

    def is_due(self):
        """Check if maintenance is due"""
        next_due = self.last_completed + timedelta(days=self.frequency_days)
        return datetime.now() >= next_due

    def days_until_due(self):
        """Days until next maintenance"""
        next_due = self.last_completed + timedelta(days=self.frequency_days)
        delta = next_due - datetime.now()
        return delta.days

    def is_overdue(self):
        """Check if maintenance is overdue"""
        return self.days_until_due() < 0

class MaintenanceManager:
    def get_upcoming_maintenance(self, days_ahead=7):
        """Get maintenance due in next N days"""
        upcoming = []
        for schedule in self.schedules:
            days_until = schedule.days_until_due()
            if 0 <= days_until <= days_ahead:
                upcoming.append({
                    'equipment': schedule.equipment_id,
                    'task': schedule.task,
                    'due_in_days': days_until,
                    'priority': 'HIGH' if days_until <= 2 else 'MEDIUM'
                })
        return sorted(upcoming, key=lambda x: x['due_in_days'])
```

---

## 11. Time Series Forecasting & Demand Prediction

### Prophet by Facebook ⭐ **RECOMMENDED**

#### Official Resources
- **Repository**: https://github.com/facebook/prophet
- **Documentation**: https://facebook.github.io/prophet/
- **Install**: `pip install prophet`

#### Why Prophet?

✅ **Pros**:
- Built for business forecasting
- Handles seasonality (daily, weekly, yearly)
- Works with missing data
- Robust to outliers
- Easy to use (intuitive API)
- No need to be a forecasting expert

❌ **Cons**:
- Large dependency (PyStan)
- Slower than simple methods
- May be overkill for simple forecasts

### Use Cases for The Lariat Bible

1. **Catering Demand Forecasting**
   - Predict busy seasons
   - Plan ingredient purchasing
   - Staff scheduling

2. **Ingredient Usage Forecasting**
   - Predict when to reorder
   - Optimize inventory levels
   - Reduce waste

3. **Revenue Forecasting**
   - Monthly revenue predictions
   - Trend analysis
   - Identify growth opportunities

### Implementation Example

```python
# modules/forecasting/demand_predictor.py

from prophet import Prophet
import pandas as pd

class DemandPredictor:
    def __init__(self):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )

    def forecast_ingredient_usage(self, ingredient_id, days_ahead=30):
        """
        Forecast ingredient usage for next N days

        Args:
            ingredient_id: Ingredient to forecast
            days_ahead: Days to forecast

        Returns:
            DataFrame with predictions
        """
        # Get historical usage data
        history = self.get_usage_history(ingredient_id)

        # Prophet requires 'ds' (date) and 'y' (value) columns
        df = pd.DataFrame({
            'ds': history['date'],
            'y': history['quantity_used']
        })

        # Fit model
        self.model.fit(df)

        # Make future dataframe
        future = self.model.make_future_dataframe(periods=days_ahead)

        # Predict
        forecast = self.model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_ahead)

    def calculate_reorder_point(self, ingredient_id, lead_time_days=7):
        """Calculate intelligent reorder point based on forecast"""
        forecast = self.forecast_ingredient_usage(ingredient_id, lead_time_days * 2)

        # Sum expected usage during lead time + safety stock
        expected_usage = forecast['yhat'].iloc[:lead_time_days].sum()
        safety_stock = forecast['yhat_upper'].iloc[:lead_time_days].sum() - expected_usage

        reorder_point = expected_usage + safety_stock

        return {
            'reorder_point': reorder_point,
            'expected_usage': expected_usage,
            'safety_stock': safety_stock,
            'lead_time_days': lead_time_days
        }
```

### Recommendations for The Lariat Bible

🎯 **Implementation Priority**:
1. **Phase 1** (Now): Simple reorder points based on average usage
2. **Phase 2** (Future): Add Prophet for seasonal forecasting
3. **Phase 3** (Advanced): ML models for demand prediction

📝 **Start simple**:
```python
# Simple average-based reorder point (implement now)
def simple_reorder_point(ingredient_id, lead_time_days=7):
    daily_avg_usage = get_average_daily_usage(ingredient_id, last_30_days=True)
    reorder_point = daily_avg_usage * lead_time_days * 1.5  # 50% safety margin
    return reorder_point
```

---

## 12. Unit Conversion for Cooking Measurements

### Pint Library ⭐ **RECOMMENDED**

#### Official Resources
- **Repository**: https://github.com/hgrecco/pint
- **Documentation**: https://pint.readthedocs.io/
- **Install**: `pip install pint`

#### Why Pint?

✅ **Features**:
- Comprehensive unit library
- Physical quantities (value + unit)
- Automatic conversions
- Arithmetic operations between units
- No dependencies
- Python 3.9+

### Cooking-Specific Alternatives

1. **Sugarcube** by vizigr0u
   - **Repository**: https://github.com/vizigr0u/sugarcube
   - **Features**: Single-file API for cooking conversions
   - **Use**: Simple, lightweight option

2. **recipe-converter** by justinmklam
   - **Repository**: https://github.com/justinmklam/recipe-converter
   - **Features**: Convert cups/tbsp/tsp to grams
   - **Use**: Imperial to metric conversion

### Implementation for The Lariat Bible

```python
# modules/recipes/unit_converter.py

from pint import UnitRegistry

ureg = UnitRegistry()

class CookingUnitConverter:
    """Handle cooking unit conversions"""

    def __init__(self):
        # Define custom cooking units
        ureg.define('cup = 236.588 * milliliter')
        ureg.define('tablespoon = tbsp = 14.7868 * milliliter')
        ureg.define('teaspoon = tsp = 4.92892 * milliliter')
        ureg.define('stick_butter = 113.398 * gram')

    def convert(self, quantity, from_unit, to_unit):
        """
        Convert between units

        Example:
            convert(2, 'cup', 'milliliter') -> 473.176
            convert(1, 'pound', 'gram') -> 453.592
        """
        amount = quantity * ureg(from_unit)
        converted = amount.to(ureg(to_unit))
        return float(converted.magnitude)

    def standardize_to_base_unit(self, quantity, unit, category='weight'):
        """
        Convert to standard base unit for storage

        Weight -> grams
        Volume -> milliliters
        """
        base_units = {
            'weight': 'gram',
            'volume': 'milliliter',
            'count': 'dimensionless'
        }

        if category in base_units:
            return self.convert(quantity, unit, base_units[category])

        return quantity

    def calculate_ingredient_cost(self, quantity, unit, price_per_case,
                                 case_size, case_unit):
        """
        Calculate ingredient cost for a recipe

        Example:
            Flour: 2 cups needed
            Case: $18.99 for 25 LB
            -> Convert 2 cups to pounds
            -> Calculate cost
        """
        # Convert recipe quantity to case units
        quantity_in_case_units = self.convert(quantity, unit, case_unit)

        # Calculate unit price
        price_per_unit = price_per_case / case_size

        # Calculate cost
        cost = quantity_in_case_units * price_per_unit

        return {
            'cost': round(cost, 2),
            'quantity_in_case_units': quantity_in_case_units,
            'unit_price': price_per_unit
        }
```

### Recommendations for The Lariat Bible

✅ **Add to requirements.txt**:
```
pint==0.23
```

✅ **Use for**:
1. Recipe ingredient conversions
2. Vendor price comparison (different pack sizes)
3. Inventory tracking (standardize units)
4. Cost calculations

---

## 13. SQLAlchemy ORM Best Practices

### Key Resources

#### 1. **sqlalchemy-orm** by MDRCS
- **Repository**: https://github.com/MDRCS/sqlalchemy-orm
- **Features**: Best practices for SQLAlchemy-CORE & ORM for large applications

#### 2. **Official SQLAlchemy**
- **Repository**: https://github.com/sqlalchemy/sqlalchemy
- **Documentation**: https://docs.sqlalchemy.org/

#### 3. **awesome-sqlalchemy** by dahlia
- **Repository**: https://github.com/dahlia/awesome-sqlalchemy
- **Features**: Curated list of SQLAlchemy tools and extensions

### Best Practices for The Lariat Bible

#### 1. Use Alembic for Migrations

```bash
# Install (already in requirements.txt ✅)
pip install alembic

# Initialize
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Add vendor_products table"

# Apply migration
alembic upgrade head
```

#### 2. Transaction Management

```python
# Use context manager for transactions
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)

def update_ingredient_price(ingredient_id, new_price):
    """Always use transactions"""
    with Session() as session:
        try:
            ingredient = session.query(Ingredient).filter_by(id=ingredient_id).first()
            ingredient.price = new_price
            ingredient.last_updated = datetime.now()

            session.commit()
            return {'success': True}
        except Exception as e:
            session.rollback()
            return {'success': False, 'error': str(e)}
```

#### 3. Relationships and Lazy Loading

```python
# modules/core/models.py

from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Recipe(Base):
    __tablename__ = 'recipes'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    category = Column(String(100))

    # Use lazy='selectin' to avoid N+1 queries
    ingredients = relationship(
        'RecipeIngredient',
        back_populates='recipe',
        lazy='selectin'
    )

class RecipeIngredient(Base):
    __tablename__ = 'recipe_ingredients'

    id = Column(Integer, primary_key=True)
    recipe_id = Column(Integer, ForeignKey('recipes.id'))
    ingredient_id = Column(Integer, ForeignKey('ingredients.id'))
    quantity = Column(Float)
    unit = Column(String(20))

    recipe = relationship('Recipe', back_populates='ingredients')
    ingredient = relationship('Ingredient')
```

#### 4. Query Optimization

```python
# Good: Use joinedload to reduce queries
from sqlalchemy.orm import joinedload

recipes = session.query(Recipe)\
    .options(joinedload(Recipe.ingredients))\
    .filter(Recipe.category == 'Catering')\
    .all()

# Good: Use select for better performance (SQLAlchemy 2.0 style)
from sqlalchemy import select

stmt = select(Recipe).where(Recipe.category == 'Catering')
recipes = session.execute(stmt).scalars().all()
```

### Recommendations for The Lariat Bible

✅ **Implement**:
1. Set up Alembic migrations
2. Use context managers for all database operations
3. Optimize relationships with appropriate lazy loading
4. Add indexes on frequently queried columns
5. Use SQLAlchemy 2.0 style queries

---

## Summary of Key Recommendations

### Immediate Actions (High Priority)

1. ✅ **Add RapidFuzz** for product matching
   ```bash
   pip install rapidfuzz
   ```

2. ✅ **Add Pint** for unit conversions
   ```bash
   pip install pint
   ```

3. ✅ **Implement Pydantic schemas** for data validation
   - Already in requirements.txt
   - Create schemas for all data structures

4. ✅ **Study invoice-parser** repository
   - https://github.com/aj-jhaveri/invoice-parser
   - Implement OCR pipeline for vendor invoices

5. ✅ **Set up Alembic** for database migrations
   - Already in requirements.txt
   - Initialize and create first migration

### Medium Priority

6. 📊 **Add invoice2data** for template-based extraction
   ```bash
   pip install invoice2data
   ```

7. 📊 **Implement fuzzy matching** for product comparisons
   - Critical for accurate vendor matching
   - See section 6 for implementation

8. 📊 **Study ETL patterns** from research
   - Data cleaning pipelines
   - Validation workflows

### Future Enhancements

9. 🔮 **Add Prophet** for demand forecasting (when ready)
   ```bash
   pip install prophet
   ```

10. 🔮 **Consider Streamlit** for advanced dashboards
    ```bash
    pip install streamlit plotly
    ```

11. 🔮 **Predictive maintenance** using ML
    - Study repositories from section 10
    - Implement when sufficient historical data

---

## Updated requirements.txt Additions

```txt
# Fuzzy matching for product comparison
rapidfuzz==3.5.2

# Unit conversion for recipes
pint==0.23

# Invoice OCR and processing
pdf2image==1.16.3
opencv-python-headless==4.8.1.78  # Headless for server deployment

# Template-based invoice extraction (optional, future)
# invoice2data==0.4.4

# Time series forecasting (optional, future)
# prophet==1.1.5

# Interactive dashboards (optional, separate app)
# streamlit==1.28.0
# plotly==5.18.0
```

---

## Repository Links Summary

### Must Review
1. **Invoice Processing**: https://github.com/aj-jhaveri/invoice-parser
2. **Fuzzy Matching**: https://github.com/rapidfuzz/RapidFuzz
3. **Pydantic**: https://github.com/pydantic/pydantic
4. **Pint**: https://github.com/hgrecco/pint
5. **SQLAlchemy Best Practices**: https://github.com/MDRCS/sqlalchemy-orm

### Good to Study
6. **Recipe Costing**: https://github.com/gmoraitis/Recipe-Cost-Calculator
7. **Restaurant Inventory**: https://github.com/shaecodes/Restaurant-Ingredient-Tracker
8. **Invoice OCR Tabular**: https://github.com/anshumyname/Invoice_ocr
9. **Prophet Forecasting**: https://github.com/facebook/prophet
10. **Streamlit Dashboards**: https://github.com/korenkaplan/Admin-dashboard

### Reference Material
11. **Excel Automation**: https://github.com/xlwings/xlwings
12. **ETL Pipelines**: https://github.com/PacktPublishing/Building-ETL-Pipelines-with-Python
13. **Restaurant Sales Analysis**: https://github.com/vltnnx/Restaurant-Sales-Analysis
14. **Price Comparison**: https://github.com/imsudip/CompareGo
15. **Awesome SQLAlchemy**: https://github.com/dahlia/awesome-sqlalchemy

---

## Next Steps

1. **Review this document** with the team
2. **Prioritize implementations** based on business needs
3. **Add selected libraries** to requirements.txt
4. **Study top repositories** for implementation patterns
5. **Create implementation tasks** in project management

---

**Document prepared for**: The Lariat Bible Development Team
**Research conducted**: 2025-11-18
**Total repositories reviewed**: 50+
**Ready to implement**: Yes 🚀
