import type { DemoData } from '../types';

export const demoData: DemoData = {
  dashboard: {
    totalReports: '20,000',
    sifPotentialReports: '4,200',
    sifPrecursorDensity: '3.8%',
    highPrecursorLocations: '12',
    densityBySite: [
      { name: 'Site A', value: 2.0, reports: 4000, sifReports: 80 },
      { name: 'Site B', value: 5.0, reports: 6000, sifReports: 300 },
      { name: 'Site C', value: 3.1, reports: 3500, sifReports: 108 },
      { name: 'Site D', value: 4.3, reports: 4500, sifReports: 193 },
      { name: 'Site E', value: 1.8, reports: 2000, sifReports: 36 },
    ],
    monthlyTrend: [
      { period: 'Jan', density: 2.1 },
      { period: 'Feb', density: 2.8 },
      { period: 'Mar', density: 3.4 },
      { period: 'Apr', density: 4.1 },
      { period: 'May', density: 3.7 },
      { period: 'Jun', density: 4.5 },
    ],
    typeDistribution: [
      { name: 'Unsafe Act', value: 8500 },
      { name: 'Unsafe Condition', value: 9200 },
      { name: 'Near Miss', value: 2100 },
      { name: 'Incident', value: 200 },
    ],
    patterns: [
      {
        id: 'P1',
        components: ['Work at Height', 'Fall Protection Failure'],
        count: 142,
        location: 'Site B, Site D',
        sifRelevance: 'High'
      },
      {
        id: 'P2',
        components: ['Maintenance', 'Pressure', 'Isolation Failure'],
        count: 89,
        location: 'Site B',
        sifRelevance: 'High'
      },
      {
        id: 'P3',
        components: ['Lifting', 'Worker Exposure', 'Line of Fire'],
        count: 76,
        location: 'Site A, Site C',
        sifRelevance: 'High'
      },
      {
        id: 'P4',
        components: ['Electrical Work', 'Energy Isolation Failure'],
        count: 54,
        location: 'Site E',
        sifRelevance: 'High'
      }
    ]
  },
  analytics: {
    densityByDepartment: [
      { name: 'Maintenance', value: 6.2 },
      { name: 'Operations', value: 4.1 },
      { name: 'Logistics', value: 2.5 },
      { name: 'Construction', value: 5.8 },
    ],
    densityByActivity: [
      { name: 'Confined Space Entry', value: 8.5 },
      { name: 'Hot Work', value: 5.2 },
      { name: 'Lifting Operations', value: 4.8 },
      { name: 'Routine Inspection', value: 0.9 },
    ],
    densityByContractor: [
      { name: 'Contractor Alpha', value: 3.4 },
      { name: 'Contractor Beta', value: 7.1 },
      { name: 'Contractor Gamma', value: 2.2 },
      { name: 'Internal Staff', value: 1.8 },
    ],
  },
  reports: [
    {
      id: 'RF-00124',
      type: 'Near Miss',
      date: '2026-05-14',
      site: 'Site B',
      department: 'Maintenance',
      contractor: 'Contractor Beta',
      text: 'During maintenance work at the wellhead, the technician was working on an energized panel. The equipment had not been isolated and the worker was exposed to the electrical hazard.',
      facts: {
        activity: 'Maintenance',
        energySource: 'Electrical',
        workerExposure: 'Direct exposure',
        criticalBarrier: 'Energy Isolation',
        barrierState: 'Missing',
        potentialConsequence: 'Severe/Fatal Electrical Injury'
      },
      sif: {
        potential: true,
        reasoning: 'High-energy electrical source + worker exposure + missing energy isolation barrier.',
        evidence: 'equipment had not been isolated',
        mappedRules: ['Energy Isolation']
      }
    },
    {
      id: 'RF-00418',
      type: 'Unsafe Act',
      date: '2026-05-21',
      site: 'Site D',
      department: 'Construction',
      contractor: 'Contractor Alpha',
      text: 'Worker observed tying off fall protection lanyard to an unapproved scaffold pipe while working at 6 meters elevation. Scaffold pipe is not rated for fall arrest loads.',
      facts: {
        activity: 'Work at Height',
        energySource: 'Gravity',
        workerExposure: 'Direct exposure',
        criticalBarrier: 'Fall Protection Anchor',
        barrierState: 'Inadequate',
        potentialConsequence: 'Fall from Height / Fatality'
      },
      sif: {
        potential: true,
        reasoning: 'Work at height > 2m + direct exposure + inadequate critical barrier (anchor point).',
        evidence: 'tying off fall protection lanyard to an unapproved scaffold pipe',
        mappedRules: ['Work at Height']
      }
    },
    {
      id: 'RF-00831',
      type: 'Unsafe Condition',
      date: '2026-05-29',
      site: 'Site A',
      department: 'Operations',
      contractor: 'Internal Staff',
      text: 'The guarding on pump P-102 was found partially removed during routine operator rounds. Pump was still running, but no workers were performing maintenance or actively exposed at the time.',
      facts: {
        activity: 'Routine Operations',
        energySource: 'Mechanical (Rotating)',
        workerExposure: 'None/Indirect',
        criticalBarrier: 'Machine Guarding',
        barrierState: 'Compromised',
        potentialConsequence: 'Minor to Severe Laceration'
      },
      sif: {
        potential: false,
        reasoning: 'Barrier compromised, but no worker exposure occurred. Risk of SIF is low without direct proximity.',
        evidence: 'no workers were performing maintenance or actively exposed',
        mappedRules: []
      }
    },
    {
      id: 'RF-01105',
      type: 'Near Miss',
      date: '2026-06-02',
      site: 'Site C',
      department: 'Logistics',
      contractor: 'Contractor Gamma',
      text: 'While lifting the 5-ton compressor unit, a rigger walked directly under the suspended load to adjust a tagline. The crane operator immediately halted the lift.',
      facts: {
        activity: 'Lifting Operations',
        energySource: 'Gravity (Suspended Load)',
        workerExposure: 'Direct exposure (Line of Fire)',
        criticalBarrier: 'Exclusion Zone',
        barrierState: 'Breached',
        potentialConsequence: 'Crush Injury / Fatality'
      },
      sif: {
        potential: true,
        reasoning: 'Suspended heavy load + direct worker exposure in line of fire + breached exclusion zone.',
        evidence: 'rigger walked directly under the suspended load',
        mappedRules: ['Line of Fire', 'Lifting Operations']
      }
    },
    {
      id: 'RF-01452',
      type: 'Unsafe Act',
      date: '2026-06-10',
      site: 'Site B',
      department: 'Maintenance',
      contractor: 'Internal Staff',
      text: 'Mechanic attempted to break open a flange on the cooling water line without verifying zero pressure. Line still had residual pressure and sprayed water, but temperature was ambient.',
      facts: {
        activity: 'Maintenance',
        energySource: 'Pressure (Low/Water)',
        workerExposure: 'Direct exposure',
        criticalBarrier: 'Pressure Verification',
        barrierState: 'Missing',
        potentialConsequence: 'Minor Injury'
      },
      sif: {
        potential: false,
        reasoning: 'Barrier failure occurred, but the energy source (low pressure, ambient temp water) lacks the capacity for Serious Injury or Fatality.',
        evidence: 'temperature was ambient',
        mappedRules: ['Energy Isolation']
      }
    }
  ],
  sifDemoReport: {
    id: 'DEMO-001',
    type: 'Near Miss',
    date: '2026-06-15',
    site: 'Site B',
    department: 'Maintenance',
    contractor: 'Internal Staff',
    text: 'During maintenance work at the wellhead, the technician was working on an energized panel. The equipment had not been isolated and the worker was exposed to the electrical hazard.',
    facts: {
      activity: 'Maintenance',
      energySource: 'Electrical',
      workerExposure: 'Direct',
      criticalBarrier: 'Energy Isolation',
      barrierState: 'Missing',
      potentialConsequence: 'Severe / Fatal Electrical Injury'
    },
    sif: {
      potential: true,
      reasoning: 'High-energy electrical source + worker exposure + missing energy isolation barrier.',
      evidence: 'equipment had not been isolated',
      mappedRules: ['Energy Isolation']
    }
  }
};
