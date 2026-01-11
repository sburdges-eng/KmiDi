-- iDAW Penta Core: Analyze Selected MIDI
-- ReaScript for Cockos Reaper
-- Version: 1.0.0
--
-- Analyzes selected MIDI items and sends to Penta Core brain server
-- Returns chord progression analysis and key detection

local PENTA_HOST = "127.0.0.1"
local PENTA_PORT = 9000

-- Get selected MIDI items
function get_selected_midi()
    local item_count = reaper.CountSelectedMediaItems(0)
    if item_count == 0 then
        reaper.ShowMessageBox("No items selected. Please select MIDI items to analyze.", "iDAW", 0)
        return nil
    end

    local midi_data = {}

    for i = 0, item_count - 1 do
        local item = reaper.GetSelectedMediaItem(0, i)
        local take = reaper.GetActiveTake(item)

        if take and reaper.TakeIsMIDI(take) then
            local retval, notecnt = reaper.MIDI_CountEvts(take)

            for n = 0, notecnt - 1 do
                local retval, selected, muted, startppqpos, endppqpos, chan, pitch, vel = reaper.MIDI_GetNote(take, n)
                if not muted then
                    table.insert(midi_data, {
                        start = startppqpos,
                        end_ = endppqpos,
                        pitch = pitch,
                        velocity = vel,
                        channel = chan
                    })
                end
            end
        end
    end

    return midi_data
end

-- Send to Penta Core via OSC
function send_to_penta(midi_data)
    if not midi_data or #midi_data == 0 then
        return false
    end

    -- Format MIDI data as JSON
    local json = "["
    for i, note in ipairs(midi_data) do
        if i > 1 then json = json .. "," end
        json = json .. string.format(
            '{"start":%d,"end":%d,"pitch":%d,"velocity":%d,"channel":%d}',
            note.start, note.end_, note.pitch, note.velocity, note.channel
        )
    end
    json = json .. "]"

    -- Send OSC message
    -- Note: Requires OSC extension or socket library
    reaper.ShowConsoleMsg("iDAW: Sending " .. #midi_data .. " notes for analysis\n")
    reaper.ShowConsoleMsg("iDAW: Data: " .. json:sub(1, 100) .. "...\n")

    -- Store data for brain server to fetch
    reaper.SetExtState("iDAW", "midi_data", json, false)
    reaper.SetExtState("iDAW", "analyze_request", os.time(), false)

    return true
end

-- Display results
function display_results()
    local chord = reaper.GetExtState("iDAW", "chord_result")
    local key = reaper.GetExtState("iDAW", "key_result")

    if chord and chord ~= "" then
        reaper.ShowConsoleMsg("iDAW Analysis Results:\n")
        reaper.ShowConsoleMsg("  Chord: " .. chord .. "\n")
        reaper.ShowConsoleMsg("  Key: " .. key .. "\n")
    end
end

-- Main
function main()
    reaper.ShowConsoleMsg("iDAW Penta Core: Analyze Selection\n")
    reaper.ShowConsoleMsg("================================\n")

    local midi_data = get_selected_midi()
    if midi_data then
        if send_to_penta(midi_data) then
            reaper.ShowConsoleMsg("iDAW: Analysis request sent to brain server\n")
            reaper.ShowConsoleMsg("iDAW: Waiting for results...\n")

            -- Poll for results (in production, use deferred callback)
            reaper.defer(function()
                display_results()
            end)
        end
    end
end

main()
