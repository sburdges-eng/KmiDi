# BEO (Banquet Event Order) Import System

## Overview

The Lariat Bible now includes a comprehensive BEO import system that allows you to import catering events and banquet orders from CSV or Excel files. This system was designed based on industry-standard BEO formats from Toast POS and event management systems used by professional catering operations.

## What is a BEO?

A **Banquet Event Order (BEO)** is a comprehensive document that contains all the details needed to execute a catering event or banquet. It serves as the single source of truth for:

- Event details (date, time, location)
- Customer information and contact details
- Guest count and seating arrangements
- Menu selections and dietary restrictions
- Pricing breakdown and payment tracking
- Special requests and setup instructions
- Staff and equipment requirements

## BEO Import Features

### Supported File Formats
- CSV (.csv)
- Excel (.xlsx, .xls)
- OpenDocument Spreadsheet (.ods)
- Google Sheets (export as CSV or Excel first)

### Data Tracked

#### Event Information
- **Event Name**: Name/title of the event
- **Event Type**: Corporate, Birthday, Wedding, etc.
- **Event Date**: Date of the event
- **Start Time**: Event start time
- **End Time**: Event end time
- **Guest Count**: Number of attendees

#### Customer Details
- **Customer Name**: Primary contact name
- **Customer Phone**: Contact phone number
- **Customer Email**: Contact email address

#### Venue Information
- **Venue Name**: Event location name
- **Venue Location**: Full address/city

#### Menu & Food
- **Menu Selection**: Selected menu or package
- **Special Requests**: Custom requests or modifications
- **Dietary Restrictions**: Allergies, vegetarian, vegan, gluten-free, etc.

#### Financial Details
- **Price Per Person**: Per-guest pricing
- **Venue Fee**: Venue rental cost
- **Service Fee**: Service charges
- **Gratuity**: Tip/gratuity amount
- **Deposit Paid**: Amount already paid
- **Subtotal**: Calculated automatically
- **Total Cost**: Calculated automatically
- **Balance Due**: Calculated automatically

#### Event Management
- **Status**: PENDING, CONFIRMED, COMPLETED, CANCELLED
- **Approved**: Boolean (0 or 1)
- **Setup Instructions**: Table arrangements, decorations, etc.
- **Equipment Needed**: A/V, microphones, projectors, etc.
- **Staff Required**: Number of staff members needed

## Usage

### 1. Generate the BEO Template

```bash
python -m modules.importers.file_importer
```

This creates `beo_catering_events_template.csv` in `data/templates/`

### 2. Fill Out the Template

Open the template in Excel or Google Sheets and fill in your event details. The template includes 3 example events:

- **Corporate Lunch** (50 guests, $2,125 total)
- **Birthday Party** (25 guests, $1,012.50 total)
- **Wedding Reception** (150 guests, $10,600 total)

### 3. Import Your BEO Data

```bash
python -m modules.importers.file_importer import beo path/to/your_events.csv
```

### 4. Review the Results

The import will show:
- ✅ Number of events imported
- 📊 Total guests across all events
- 💰 Total revenue
- ⚠️ Any errors encountered

## Example Output

```
📋 Importing BEO/Catering Event from data/templates/beo_catering_events_template.csv
✅ Imported 3 BEO/Catering events
   Total guests: 225
   Total revenue: $13,737.50

✅ Import successful!
```

## Field Requirements

### Required Fields (Must be present):
- `event_name`
- `event_date`
- `guest_count`
- `customer_name`
- `customer_phone`

### Optional Fields (Will use defaults if missing):
- `event_type` (default: "Catering")
- `start_time` (default: "12:00 PM")
- `end_time` (default: "4:00 PM")
- `customer_email` (default: empty)
- `venue_name` (default: "The Lariat")
- `venue_location` (default: "Fort Collins, CO")
- `menu_selection` (default: empty)
- `special_requests` (default: empty)
- `dietary_restrictions` (default: empty)
- `price_per_person` (default: 0.00)
- `venue_fee` (default: 0.00)
- `service_fee` (default: 0.00)
- `gratuity` (default: 0.00)
- `deposit_paid` (default: 0.00)
- `status` (default: "PENDING")
- `approved` (default: 0)
- `setup_instructions` (default: empty)
- `equipment_needed` (default: empty)
- `staff_required` (default: 0)

## Automatic Calculations

The system automatically calculates:

```python
Subtotal = (Price Per Person × Guest Count) + Venue Fee
Total Cost = Subtotal + Service Fee + Gratuity
Balance Due = Total Cost - Deposit Paid
```

## Event Status Values

- **PENDING**: Event inquiry, not yet confirmed
- **CONFIRMED**: Event booked and confirmed
- **COMPLETED**: Event has occurred
- **CANCELLED**: Event was cancelled

## Integration with The Lariat System

### Revenue Tracking
BEO imports integrate with The Lariat's financial tracking:
- Monthly catering revenue: Track against $28,000 target
- 45% target margin for catering events
- Automatic calculation of food costs vs. revenue

### Vendor Integration
Menu selections can be linked to:
- Recipe costing system (modules/recipes/)
- Vendor pricing (Shamrock Foods vs SYSCO)
- Ingredient availability

### Equipment Management
Equipment requirements tracked in:
- modules/equipment/equipment_manager.py
- Maintenance scheduling
- Availability tracking

### Staff Scheduling
Staff requirements feed into:
- Labor cost calculations
- Scheduling system
- Capacity planning

## Exporting BEO Data

Once imported, you can export events to:
- PDF for customer delivery
- Excel for analysis
- Google Sheets for collaborative planning

## Google Sheets Integration

### Option 1: Export from Google Sheets
1. Open your BEO data in Google Sheets
2. File → Download → Comma-separated values (.csv)
3. Import the CSV using the BEO import command

### Option 2: Google Sheets API (Future)
Direct integration with Google Sheets API will allow:
- Real-time sync of event data
- Automatic updates when sheets are modified
- Collaborative editing with staff

## Best Practices

### 1. Consistent Naming
- Use clear, descriptive event names
- Include date or customer name for easy identification

### 2. Accurate Guest Counts
- Update guest counts as they change
- Track confirmed vs. estimated counts

### 3. Detailed Menu Selections
- Reference specific menu IDs or recipe names
- Document all customizations

### 4. Complete Dietary Information
- List all allergies and restrictions
- Include quantities (e.g., "5 vegetarian")

### 5. Financial Tracking
- Record all deposits immediately
- Update totals when pricing changes
- Track balance due for collections

### 6. Setup Documentation
- Document table arrangements clearly
- List all equipment needs specifically
- Note timing for setup and teardown

## Error Handling

Common import errors and solutions:

### Missing Required Fields
```
Error: Missing columns: ['customer_phone']
Solution: Ensure all required fields are present in your CSV
```

### Invalid Data Types
```
Error: Row 3: invalid literal for int() with base 10: 'fifty'
Solution: Use numeric values for guest_count, prices, etc.
```

### Date Format Issues
```
Best format: YYYY-MM-DD (2025-02-15)
Also accepts: MM/DD/YYYY, DD/MM/YYYY
```

## Tips for Importing from Toast POS

If you're exporting from Toast POS:

1. **Event Name** = Event/Booking Name
2. **Guest Count** = Party Size
3. **Customer Info** = Contact Information
4. **Menu Selection** = Package/Menu Selected
5. **Pricing** = May need manual calculation from Toast reports

## Tips for Importing from Google Drive

If you have existing BEOs in Google Drive:

1. **Organize** events in a single spreadsheet
2. **Standardize** column names to match template
3. **Export** to CSV or Excel format
4. **Import** using the BEO import command
5. **Verify** imported data for accuracy

## Example BEO CSV Format

```csv
Event Name,Event Type,Event Date,Guest Count,Customer Name,Customer Phone
"Company Holiday Party",Corporate,2025-12-15,75,"Jane Smith","970-555-1111"
"50th Birthday Bash",Birthday,2025-06-20,40,"Bob Johnson","970-555-2222"
"Graduation Celebration",Graduation,2025-05-18,60,"Maria Garcia","970-555-3333"
```

## Programmatic Access

### Python Integration

```python
from modules.importers.file_importer import FileImporter

importer = FileImporter()

# Import BEO data
result = importer.import_beo('data/my_events.csv')

# Access imported events
for event in result['events']:
    print(f"{event['event_name']}: ${event['total_cost']:,.2f}")

# Get summary statistics
print(f"Total Revenue: ${result['total_revenue']:,.2f}")
print(f"Total Guests: {result['total_guests']}")
```

### Return Value Structure

```python
{
    'success': True,
    'total_rows': 3,
    'events_imported': 3,
    'errors': [],
    'error_count': 0,
    'events': [
        {
            'event_name': 'Corporate Lunch',
            'guest_count': 50,
            'total_cost': 2125.00,
            'balance_due': 1625.00,
            # ... all other fields
        },
        # ... more events
    ],
    'total_revenue': 13737.50,
    'total_guests': 225
}
```

## Future Enhancements

Planned features for the BEO system:

1. **Email Notifications**
   - Automatic confirmation emails to customers
   - Staff alerts for upcoming events
   - Payment reminders for balance due

2. **Calendar Integration**
   - Sync events to Google Calendar
   - Venue conflict detection
   - Automated reminders

3. **Recipe/Menu Linking**
   - Automatic menu costing from recipes
   - Ingredient availability checking
   - Vendor price optimization

4. **Staff Assignment**
   - Automatic staffing recommendations
   - Availability checking
   - Labor cost calculations

5. **Reporting Dashboard**
   - Monthly revenue summaries
   - Event type analysis
   - Customer insights

## Support

For issues or questions:
- Check error messages carefully
- Verify CSV formatting matches template
- Ensure all required fields are present
- Contact system administrator if problems persist

## Schema Reference

For database developers, the BEO import creates records compatible with:

```sql
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY,
    event_name TEXT NOT NULL,
    event_type TEXT,
    event_date DATE NOT NULL,
    start_time TIME,
    end_time TIME,
    guest_count INTEGER NOT NULL,
    customer_name TEXT NOT NULL,
    customer_phone TEXT NOT NULL,
    customer_email TEXT,
    venue_name TEXT,
    venue_location TEXT,
    menu_selection TEXT,
    special_requests TEXT,
    dietary_restrictions TEXT,
    price_per_person DECIMAL(10,2),
    venue_fee DECIMAL(10,2),
    service_fee DECIMAL(10,2),
    gratuity DECIMAL(10,2),
    deposit_paid DECIMAL(10,2),
    subtotal DECIMAL(10,2),
    total_cost DECIMAL(10,2),
    balance_due DECIMAL(10,2),
    status TEXT,
    approved INTEGER,
    setup_instructions TEXT,
    equipment_needed TEXT,
    staff_required INTEGER,
    created_date TIMESTAMP,
    last_modified TIMESTAMP
);
```

## Conclusion

The BEO import system streamlines event management for The Lariat by:
- ✅ Standardizing event data capture
- ✅ Automating financial calculations
- ✅ Integrating with existing systems
- ✅ Supporting multiple data sources
- ✅ Providing clear error handling

Start using it today to manage your catering events more efficiently!
