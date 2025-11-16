export interface AttendanceRecord {
    name: string;
    roll: string;
    time: string;
}

export interface User {
    name: string;
    roll: string;
    identifier: string;
}

export interface Settings {
    auto_attendance_interval_minutes: number;
    nimgs: number;
    pipeline_timeout: number;
    stable_time: number;
    max_center_movement: number;
    match_distance_threshold: number;
}

export interface AppData {
    attendance: AttendanceRecord[];
    users: User[];
    settings: Settings;
    totalreg: number;
    datetoday2: string;
    auto_attendance: boolean;
    recognition_active?: boolean;
    enrollment_active?: boolean;
}

