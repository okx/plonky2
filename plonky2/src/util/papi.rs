use papi_bindings::counter::Counter;
use papi_bindings::events_set::EventsSet;

pub fn init_papi() -> EventsSet {
    papi_bindings::initialize(true).unwrap();
    let counters = vec![Counter::from_name("instructions").unwrap()];
    EventsSet::new(&counters).unwrap()
}

pub fn stop_papi(event_set: &mut EventsSet, msg: &str) {
    let counters = event_set.stop().unwrap();
    println!("No. of instructions {}: {}", msg, counters[0]);
}
