"""
Equipment Management Module
Tracks kitchen equipment, maintenance schedules, and service history
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

class EquipmentStatus(Enum):
    """Equipment operational status"""
    OPERATIONAL = "Operational"
    NEEDS_MAINTENANCE = "Needs Maintenance"
    UNDER_REPAIR = "Under Repair"
    OUT_OF_SERVICE = "Out of Service"
    RETIRED = "Retired"

class MaintenanceType(Enum):
    """Types of maintenance activities"""
    DAILY_CLEANING = "Daily Cleaning"
    WEEKLY_CLEANING = "Weekly Cleaning"
    MONTHLY_INSPECTION = "Monthly Inspection"
    QUARTERLY_SERVICE = "Quarterly Service"
    ANNUAL_SERVICE = "Annual Service"
    REPAIR = "Repair"
    EMERGENCY = "Emergency Repair"

@dataclass
class Equipment:
    """Kitchen equipment with full tracking"""
    
    # Basic Information
    equipment_id: str
    name: str
    category: str  # Cooking, Refrigeration, Prep, Dishwashing, etc.
    brand: str
    model: str
    serial_number: str
    
    # Location
    location: str  # Kitchen, Prep Area, Bar, Storage, etc.
    station: Optional[str] = None  # Grill, Fryer, Cold Line, etc.
    
    # Purchase Information
    purchase_date: datetime = None
    purchase_price: float = 0.0
    vendor: str = ""
    warranty_end_date: Optional[datetime] = None
    
    # Specifications
    specifications: Dict = None  # Capacity, power requirements, dimensions, etc.
    
    # Maintenance Schedule
    daily_tasks: List[str] = None
    weekly_tasks: List[str] = None
    monthly_tasks: List[str] = None
    quarterly_tasks: List[str] = None
    annual_tasks: List[str] = None
    
    # Service Information
    service_company: str = ""
    service_contact: str = ""
    service_phone: str = ""
    service_contract: bool = False
    
    # Current Status
    status: EquipmentStatus = EquipmentStatus.OPERATIONAL
    last_maintenance_date: Optional[datetime] = None
    next_maintenance_due: Optional[datetime] = None
    
    # Operating Instructions
    operating_instructions: List[str] = None
    safety_guidelines: List[str] = None
    
    # Parts Information
    common_parts: List[Dict] = None  # Part name, number, supplier, cost
    
    def __post_init__(self):
        """Initialize defaults"""
        if self.specifications is None:
            self.specifications = {}
        if self.daily_tasks is None:
            self.daily_tasks = []
        if self.weekly_tasks is None:
            self.weekly_tasks = []
        if self.monthly_tasks is None:
            self.monthly_tasks = []
        if self.quarterly_tasks is None:
            self.quarterly_tasks = []
        if self.annual_tasks is None:
            self.annual_tasks = []
        if self.operating_instructions is None:
            self.operating_instructions = []
        if self.safety_guidelines is None:
            self.safety_guidelines = []
        if self.common_parts is None:
            self.common_parts = []
    
    @property
    def age_years(self) -> float:
        """Calculate equipment age in years"""
        if self.purchase_date:
            age = datetime.now() - self.purchase_date
            return age.days / 365.25
        return 0
    
    @property
    def warranty_status(self) -> str:
        """Check warranty status"""
        if self.warranty_end_date:
            if datetime.now() < self.warranty_end_date:
                days_left = (self.warranty_end_date - datetime.now()).days
                return f"Active - {days_left} days remaining"
            return "Expired"
        return "No warranty information"
    
    @property
    def depreciated_value(self) -> float:
        """Calculate depreciated value (straight-line, 7-year for equipment)"""
        if self.purchase_price and self.purchase_date:
            depreciation_years = 7
            annual_depreciation = self.purchase_price / depreciation_years
            current_value = self.purchase_price - (annual_depreciation * self.age_years)
            return max(0, current_value)  # Don't go negative
        return 0
    
    def is_maintenance_due(self) -> bool:
        """Check if maintenance is due"""
        if self.next_maintenance_due:
            return datetime.now() >= self.next_maintenance_due
        return False
    
    def get_maintenance_checklist(self, maintenance_type: MaintenanceType) -> List[str]:
        """Get checklist for specific maintenance type"""
        checklists = {
            MaintenanceType.DAILY_CLEANING: self.daily_tasks,
            MaintenanceType.WEEKLY_CLEANING: self.weekly_tasks,
            MaintenanceType.MONTHLY_INSPECTION: self.monthly_tasks,
            MaintenanceType.QUARTERLY_SERVICE: self.quarterly_tasks,
            MaintenanceType.ANNUAL_SERVICE: self.annual_tasks
        }
        return checklists.get(maintenance_type, [])


@dataclass
class MaintenanceRecord:
    """Record of maintenance performed"""
    
    record_id: str
    equipment_id: str
    date_performed: datetime
    maintenance_type: MaintenanceType
    performed_by: str
    
    # Details
    tasks_completed: List[str] = None
    issues_found: List[str] = None
    parts_replaced: List[Dict] = None  # Part name, cost
    
    # Costs
    labor_hours: float = 0.0
    labor_cost: float = 0.0
    parts_cost: float = 0.0
    
    # Next maintenance
    next_maintenance_date: Optional[datetime] = None
    
    # Notes
    notes: str = ""
    
    def __post_init__(self):
        """Initialize defaults"""
        if self.tasks_completed is None:
            self.tasks_completed = []
        if self.issues_found is None:
            self.issues_found = []
        if self.parts_replaced is None:
            self.parts_replaced = []
    
    @property
    def total_cost(self) -> float:
        """Calculate total maintenance cost"""
        return self.labor_cost + self.parts_cost


class EquipmentManager:
    """Manage all equipment and maintenance"""
    
    def __init__(self):
        self.equipment_list: List[Equipment] = []
        self.maintenance_history: List[MaintenanceRecord] = []
    
    def add_equipment(self, equipment: Equipment) -> str:
        """Add new equipment to inventory"""
        self.equipment_list.append(equipment)
        return f"Added {equipment.name} to equipment inventory"
    
    def get_maintenance_schedule(self, days_ahead: int = 30) -> List[Dict]:
        """Get upcoming maintenance schedule"""
        schedule = []
        end_date = datetime.now() + timedelta(days=days_ahead)
        
        for equipment in self.equipment_list:
            if equipment.next_maintenance_due:
                if equipment.next_maintenance_due <= end_date:
                    schedule.append({
                        'equipment': equipment.name,
                        'location': equipment.location,
                        'due_date': equipment.next_maintenance_due,
                        'status': equipment.status.value,
                        'days_until_due': (equipment.next_maintenance_due - datetime.now()).days,
                        'overdue': equipment.is_maintenance_due()
                    })
        
        # Sort by due date
        schedule.sort(key=lambda x: x['due_date'])
        return schedule
    
    def record_maintenance(self, record: MaintenanceRecord) -> str:
        """Record completed maintenance"""
        self.maintenance_history.append(record)
        
        # Update equipment status
        equipment = self.get_equipment_by_id(record.equipment_id)
        if equipment:
            equipment.last_maintenance_date = record.date_performed
            equipment.next_maintenance_due = record.next_maintenance_date
            if equipment.status == EquipmentStatus.NEEDS_MAINTENANCE:
                equipment.status = EquipmentStatus.OPERATIONAL
        
        return f"Maintenance recorded for equipment {record.equipment_id}"
    
    def get_equipment_by_id(self, equipment_id: str) -> Optional[Equipment]:
        """Find equipment by ID"""
        for eq in self.equipment_list:
            if eq.equipment_id == equipment_id:
                return eq
        return None
    
    def get_equipment_by_location(self, location: str) -> List[Equipment]:
        """Get all equipment in a specific location"""
        return [eq for eq in self.equipment_list if eq.location == location]
    
    def get_maintenance_costs(self, start_date: datetime, end_date: datetime) -> Dict:
        """Calculate maintenance costs for a period"""
        period_records = [
            r for r in self.maintenance_history
            if start_date <= r.date_performed <= end_date
        ]
        
        total_labor = sum(r.labor_cost for r in period_records)
        total_parts = sum(r.parts_cost for r in period_records)
        
        # Group by equipment
        by_equipment = {}
        for record in period_records:
            if record.equipment_id not in by_equipment:
                by_equipment[record.equipment_id] = {
                    'count': 0,
                    'labor_cost': 0,
                    'parts_cost': 0,
                    'total_cost': 0
                }
            
            by_equipment[record.equipment_id]['count'] += 1
            by_equipment[record.equipment_id]['labor_cost'] += record.labor_cost
            by_equipment[record.equipment_id]['parts_cost'] += record.parts_cost
            by_equipment[record.equipment_id]['total_cost'] += record.total_cost
        
        return {
            'period': f"{start_date.date()} to {end_date.date()}",
            'total_maintenance_events': len(period_records),
            'total_labor_cost': total_labor,
            'total_parts_cost': total_parts,
            'total_cost': total_labor + total_parts,
            'by_equipment': by_equipment,
            'average_cost_per_event': (total_labor + total_parts) / len(period_records) if period_records else 0
        }
    
    def get_equipment_summary(self) -> Dict:
        """Get summary of all equipment"""
        total_value = sum(eq.purchase_price for eq in self.equipment_list)
        depreciated_value = sum(eq.depreciated_value for eq in self.equipment_list)
        
        status_counts = {}
        for eq in self.equipment_list:
            status = eq.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_equipment': len(self.equipment_list),
            'original_value': total_value,
            'depreciated_value': depreciated_value,
            'depreciation': total_value - depreciated_value,
            'status_breakdown': status_counts,
            'maintenance_due': len([eq for eq in self.equipment_list if eq.is_maintenance_due()]),
            'under_warranty': len([eq for eq in self.equipment_list if eq.warranty_status.startswith('Active')])
        }
