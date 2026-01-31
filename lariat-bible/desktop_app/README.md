# Lariat Bible Desktop Application

A comprehensive restaurant management system for The Lariat restaurant in Fort Collins, Colorado.

## Features

### 📊 Dashboard
- Real-time revenue metrics ($48K monthly revenue)
- Catering vs Restaurant revenue split (58% / 42%)
- Food cost and labor cost tracking
- Vendor savings tracking ($52K annual savings with Shamrock)
- Quick action buttons for common tasks
- Recent activity feed

### 📖 Recipe Management
- Complete recipe library with categories
- Ingredient tracking and costing
- Recipe cost analysis (food cost %, profit margins)
- Nutritional information
- Instructions and preparation notes
- Search and filter capabilities

### 📦 Inventory Management
- Real-time inventory tracking
- Par level management
- Low stock alerts
- Category-based organization (Produce, Meat, Dairy, etc.)
- Inventory value tracking ($8,450 current value)
- Receiving and counting functions

### 📄 Invoice Management
- Invoice import and OCR scanning
- Vendor price comparison (SYSCO vs Shamrock)
- 29.5% average savings tracking
- Historical invoice storage
- Automated price analysis

### 🎯 Catering Module
- Event calendar and list views
- BEO (Banquet Event Order) generation
- Quote creation
- Client management
- Profit tracking (45% margin)
- $28K monthly catering revenue tracking

### 🛒 Ordering System
- Automated order generation
- Par level-based ordering
- Multi-vendor support
- Shopping list generation
- Order history tracking
- Cost optimization

### 🔧 Equipment Maintenance
- Equipment tracking and scheduling
- Maintenance history
- Service reminders
- Compliance tracking (92% rate)
- Schedule printing

### 📈 Analytics & Reporting
- Revenue analysis
- Cost breakdown
- Menu performance metrics
- Catering analysis
- Daily/Weekly/Monthly reports
- Export to PDF and Excel

## Installation

1. Ensure Python 3.7+ is installed
2. Install required dependencies:
```bash
cd /Users/seanburdges/lariat-bible
pip install -r requirements.txt
```

3. If tkinter is not installed (comes with most Python installations):
```bash
# On macOS:
brew install python-tk

# On Ubuntu/Debian:
sudo apt-get install python3-tk

# On Windows: tkinter is included with Python
```

## Running the Application

### Method 1: Using the launch script
```bash
cd /Users/seanburdges/lariat-bible/desktop_app
python3 launch.py
```

### Method 2: Direct execution
```bash
cd /Users/seanburdges/lariat-bible/desktop_app
python3 main.py
```

### Method 3: Make executable and run
```bash
cd /Users/seanburdges/lariat-bible/desktop_app
chmod +x launch.py
./launch.py
```

## Key Metrics Displayed

- **Monthly Revenue**: $48,000
  - Catering: $28,000 (58%)
  - Restaurant: $20,000 (42%)

- **Cost Structure**:
  - Food Cost: 28.5%
  - Labor Cost: 32%
  - Prime Cost: 60.5%

- **Vendor Savings** (Shamrock vs SYSCO):
  - Monthly: $4,333
  - Annual: $52,000
  - Percentage: 29.5%

## Navigation

The application uses a tabbed interface with 8 main sections:

1. **Dashboard** - Overview and quick actions
2. **Recipes** - Recipe management and costing
3. **Inventory** - Stock tracking and management
4. **Invoices** - Invoice processing and vendor comparison
5. **Catering** - Event management and BEOs
6. **Ordering** - Purchase orders and shopping lists
7. **Equipment** - Maintenance tracking
8. **Analytics** - Reports and business intelligence

## Menu Bar

- **File**: New/Open/Save recipes, Import invoices, Export reports
- **Edit**: Settings, Vendor management
- **View**: Quick navigation to different tabs
- **Reports**: Generate various reports
- **Help**: Documentation and about

## Data Storage

The application integrates with the existing lariat-bible data structure:
- Recipes stored in YAML format
- Inventory data in structured files
- Invoice images for OCR processing
- Excel templates for various functions

## Support

For issues or questions about the Lariat Bible Desktop Application, please refer to the main project documentation or contact the development team.

## Version

Current Version: 1.0.0

## License

Proprietary - For use by The Lariat restaurant only
