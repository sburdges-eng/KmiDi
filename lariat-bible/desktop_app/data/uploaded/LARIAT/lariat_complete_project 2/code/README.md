# The Lariat Bible 🤠

## Comprehensive Restaurant Management System

A unified platform for managing all aspects of The Lariat restaurant operations in Fort Collins, Colorado.

## 🎯 Project Vision

The Lariat Bible serves as the single source of truth for:
- Vendor pricing and analysis
- Inventory management
- Recipe standardization and costing
- Catering operations
- Equipment maintenance
- Financial reporting
- Staff training and documentation

## 📊 Key Metrics
- **Monthly Catering Revenue**: $28,000
- **Monthly Restaurant Revenue**: $20,000
- **Potential Annual Savings** (Shamrock vs SYSCO): $52,000

## 🗂️ Project Structure

```
lariat-bible/
├── core/                   # Core functionality
│   ├── database/          # Database models and connections
│   ├── authentication/    # User authentication and permissions
│   └── shared_utilities/  # Shared helper functions
├── modules/               # Business logic modules
│   ├── vendor_analysis/   # Vendor price comparison and optimization
│   ├── inventory/         # Stock management and tracking
│   ├── recipes/           # Recipe management and costing
│   ├── catering/          # Catering operations and quotes
│   ├── maintenance/       # Equipment maintenance schedules
│   └── reporting/         # Business intelligence and reports
├── web_interface/         # Web application frontend
├── data/                  # Data storage
│   └── invoices/         # Invoice images and OCR data
├── documentation/         # Additional documentation
└── tests/                # Test suites
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/lariat-bible.git
cd lariat-bible

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run initial setup
python setup.py
```

### Basic Usage

```python
# Example: Vendor price comparison
from modules.vendor_analysis import VendorComparator

comparator = VendorComparator()
savings = comparator.compare_vendors('SYSCO', 'Shamrock Foods')
print(f"Potential monthly savings: ${savings}")
```

## 📦 Modules Overview

### Vendor Analysis
Automated price comparison between vendors with OCR invoice processing.
- Invoice OCR and data extraction
- Price trend analysis
- Savings opportunity identification

### Inventory Management
Real-time inventory tracking and automated reordering.
- Stock level monitoring
- Expiration date tracking
- Automated purchase order generation

### Recipe Management
Standardized recipes with automatic cost calculation.
- Recipe scaling for different serving sizes
- Ingredient cost tracking
- Margin analysis

### Catering Operations
Streamlined catering workflow from quote to execution.
- Quick quote generator
- Event planning tools
- Profit margin calculator (Target: 45%)

### Maintenance Tracking
Equipment maintenance scheduling and history.
- Preventive maintenance schedules
- Repair history logging
- Vendor contact management

### Reporting Dashboard
Comprehensive business intelligence and analytics.
- Daily/weekly/monthly sales reports
- Labor cost analysis
- Profit margin tracking

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```env
DATABASE_URL=sqlite:///lariat.db
SECRET_KEY=your-secret-key
INVOICE_STORAGE_PATH=./data/invoices
```

## 📈 Development Roadmap

### Phase 1: Foundation (Current)
- [x] Project structure setup
- [ ] Database schema design
- [ ] Core utilities implementation

### Phase 2: Vendor Analysis
- [ ] OCR pipeline for invoices
- [ ] Price comparison engine
- [ ] Savings report generator

### Phase 3: Inventory & Recipes
- [ ] Inventory tracking system
- [ ] Recipe database
- [ ] Cost calculation engine

### Phase 4: Web Interface
- [ ] Dashboard creation
- [ ] Mobile-responsive design
- [ ] Real-time updates

## 🤝 Contributing

This is a private repository for The Lariat restaurant operations.

## 📝 License

Proprietary - The Lariat Restaurant, Fort Collins, CO

## 👤 Owner

**Sean** - Restaurant Owner & Operator

## 🆘 Support

For questions or issues, contact Sean directly.

---

*Building a data-driven future for The Lariat, one module at a time.*
