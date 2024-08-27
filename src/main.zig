const std = @import("std");
const Value = @import("engine.zig").Value;

pub fn main() !void {
    std.debug.print("Starting main function\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) @panic("Memory leak detected");
    }
    const allocator = gpa.allocator();

    std.debug.print("Creating Value a\n", .{});
    const a = try Value.init(allocator, 2.0);
    defer a.deinit();
    std.debug.print("Creating Value b\n", .{});
    const b = try Value.init(allocator, -3.0);
    defer b.deinit();
    std.debug.print("Creating Value c\n", .{});
    const c = try Value.init(allocator, 10.0);
    defer c.deinit();

    std.debug.print("Multiplying a and b\n", .{});
    const e = try a.mul(b);
    defer e.deinit();
    std.debug.print("Adding e and c\n", .{});
    const d = try e.add(c);
    defer d.deinit();
    std.debug.print("Applying ReLU to d\n", .{});
    const f = try d.relu();
    defer f.deinit();

    std.debug.print("Starting backward pass\n", .{});
    try f.backward();

    std.debug.print("a = {}\n", .{a.*});
    std.debug.print("b = {}\n", .{b.*});
    std.debug.print("c = {}\n", .{c.*});
    std.debug.print("d = {}\n", .{d.*});
    std.debug.print("e = {}\n", .{e.*});
    std.debug.print("f = {}\n", .{f.*});
    std.debug.print("Finished main function\n", .{});
}
