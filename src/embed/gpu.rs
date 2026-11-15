//! Which graphics card DirectML should use.

/// A hardware graphics adapter as DXGI lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Adapter {
    /// Its position in DXGI's adapter list, which is DirectML's device id.
    pub index: u32,
    pub name: String,
    pub vram_mb: u64,
}

/// The hardware adapter with the most dedicated video memory: on a laptop with integrated
/// and discrete graphics that is the discrete card, which DXGI does not always list first.
/// `None` when there is none (only Windows' software renderer).
#[cfg(windows)]
pub fn adapter() -> Option<Adapter> {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1};
    // SAFETY: plain COM calls; the interfaces release themselves when dropped.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut adapters = Vec::new();
        for index in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(index) else { break };
            let Ok(description) = adapter.GetDesc1() else { continue };
            if description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }
            let length = description.Description.iter().position(|&unit| unit == 0).unwrap_or(description.Description.len());
            let name = String::from_utf16_lossy(&description.Description[..length]).trim().to_owned();
            adapters.push(Adapter { index, name, vram_mb: description.DedicatedVideoMemory as u64 / (1024 * 1024) });
        }
        best(adapters)
    }
}

#[cfg(not(windows))]
pub fn adapter() -> Option<Adapter> {
    None
}

/// The first of the adapters with the most memory.
fn best(adapters: Vec<Adapter>) -> Option<Adapter> {
    adapters.into_iter().rev().max_by_key(|adapter| adapter.vram_mb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(index: u32, vram_mb: u64) -> Adapter {
        Adapter { index, name: format!("card {index}"), vram_mb }
    }

    #[test]
    fn prefers_the_card_with_the_most_memory_then_the_first() {
        assert_eq!(best(vec![card(0, 128), card(1, 8192)]), Some(card(1, 8192)));
        assert_eq!(best(vec![card(0, 4096), card(1, 4096)]), Some(card(0, 4096)));
        assert_eq!(best(Vec::new()), None);
    }
}
