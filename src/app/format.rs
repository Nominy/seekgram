//! Numbers as people read them: grouped counts, sizes in sensible units, waits in
//! minutes. The panel never shows raw bytes or seconds.

/// `1234567` becomes `1,234,567`.
pub fn count(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

/// A size on disk or in transit: `2.9 GB`, `512 MB`, `14 KB`.
pub fn bytes(value: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;
    let size = value as f64;
    if size >= TB {
        format!("{:.1} TB", size / TB)
    } else if size >= GB {
        format!("{:.1} GB", size / GB)
    } else if size >= 10.0 * MB {
        format!("{:.0} MB", size / MB)
    } else if size >= MB {
        format!("{:.1} MB", size / MB)
    } else if value == 0 {
        "0 KB".to_owned()
    } else {
        format!("{:.0} KB", (size / KB).max(1.0))
    }
}

/// How long a wait is, to the minute: `less than a minute`, `3 min`, `1 h 5 min`.
pub fn wait(seconds: u64) -> String {
    if seconds < 60 {
        return "less than a minute".to_owned();
    }
    let minutes = seconds.div_ceil(60);
    match (minutes / 60, minutes % 60) {
        (0, minutes) => format!("{minutes} min"),
        (hours, 0) => format!("{hours} h"),
        (hours, minutes) => format!("{hours} h {minutes} min"),
    }
}

/// Items per minute. Slow rates keep one decimal so they do not read as zero.
pub fn rate(per_minute: f64) -> String {
    if !per_minute.is_finite() || per_minute <= 0.0 {
        "0".to_owned()
    } else if per_minute < 10.0 {
        format!("{per_minute:.1}")
    } else {
        count(per_minute.round() as u64)
    }
}

/// `done` of `total` as a percentage for a progress bar, 0 when the total is unknown.
pub fn percent(done: u64, total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    (done.min(total) as f64 / total as f64 * 100.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_are_grouped_in_threes() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(1234), "1,234");
        assert_eq!(count(1234567), "1,234,567");
    }

    #[test]
    fn sizes_pick_a_readable_unit() {
        assert_eq!(bytes(0), "0 KB");
        assert_eq!(bytes(300), "1 KB");
        assert_eq!(bytes(14 * 1024), "14 KB");
        assert_eq!(bytes(5 * 1024 * 1024 / 2), "2.5 MB");
        assert_eq!(bytes(512 * 1024 * 1024), "512 MB");
        assert_eq!(bytes(3_113_851_290), "2.9 GB");
    }

    #[test]
    fn waits_are_in_minutes_and_hours() {
        assert_eq!(wait(12), "less than a minute");
        assert_eq!(wait(170), "3 min");
        assert_eq!(wait(3600), "1 h");
        assert_eq!(wait(3900), "1 h 5 min");
    }

    #[test]
    fn rates_and_percentages_stay_in_range() {
        assert_eq!(rate(0.0), "0");
        assert_eq!(rate(2.34), "2.3");
        assert_eq!(rate(1499.6), "1,500");
        assert_eq!(percent(5, 0), 0.0);
        assert_eq!(percent(50, 200), 25.0);
        assert_eq!(percent(300, 200), 100.0);
    }
}
