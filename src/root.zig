const std = @import("std");
const testing = std.testing;
const Value = @import("engine.zig").Value;

pub fn main() !void {
    std.debug.print("Running no tests...\n", .{});
    testing.refAllDecls(@This());
}
