const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

pub const Value = struct {
    data: f64,
    grad: f64,
    prev: std.ArrayList(*Value),
    op: []const u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, data: f64) !*Value {
        std.debug.print("Initializing Value with data: {d}\n", .{data});
        const self = try allocator.create(Value);
        self.* = .{
            .data = data,
            .grad = 0,
            .prev = std.ArrayList(*Value).init(allocator),
            .op = "",
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Value) void {
        std.debug.print("Deinitializing Value with data: {d}\n", .{self.data});
        self.prev.deinit();
        self.allocator.destroy(self);
    }

    pub fn add(self: *Value, other: *Value) !*Value {
        const out = try Value.init(self.allocator, self.data + other.data);
        try out.prev.appendSlice(&[_]*Value{ self, other });
        out.op = "+";
        return out;
    }

    pub fn mul(self: *Value, other: *Value) !*Value {
        const out = try Value.init(self.allocator, self.data * other.data);
        try out.prev.appendSlice(&[_]*Value{ self, other });
        out.op = "*";
        return out;
    }

    pub fn pow(self: *Value, exponent: f64) !*Value {
        const out = try Value.init(self.allocator, math.pow(f64, self.data, exponent));
        try out.prev.append(self);
        out.op = try std.fmt.allocPrint(self.allocator, "**{d}", .{exponent});
        return out;
    }

    pub fn relu(self: *Value) !*Value {
        const out = try Value.init(self.allocator, if (self.data < 0) 0 else self.data);
        try out.prev.append(self);
        out.op = "ReLU";
        return out;
    }

    pub fn backward(self: *Value) !void {
        std.debug.print("Starting backward pass\n", .{});
        var topo = std.ArrayList(*Value).init(self.allocator);
        defer topo.deinit();

        var visited = std.AutoHashMap(*Value, void).init(self.allocator);
        defer visited.deinit();

        try buildTopo(self, &topo, &visited);

        self.grad = 1;
        var i: usize = topo.items.len;
        while (i > 0) : (i -= 1) {
            std.debug.print("Backward pass for node: {}\n", .{topo.items[i - 1].*});
            try topo.items[i - 1].backwardOp();
        }
        std.debug.print("Finished backward pass\n", .{});
    }

    fn backwardOp(self: *Value) !void {
        if (std.mem.eql(u8, self.op, "+")) {
            self.prev.items[0].grad += self.grad;
            self.prev.items[1].grad += self.grad;
        } else if (std.mem.eql(u8, self.op, "*")) {
            self.prev.items[0].grad += self.prev.items[1].data * self.grad;
            self.prev.items[1].grad += self.prev.items[0].data * self.grad;
        } else if (std.mem.startsWith(u8, self.op, "**")) {
            const base = self.prev.items[0];
            const exponent = try std.fmt.parseFloat(f64, self.op[2..]);
            base.grad += exponent * math.pow(f64, base.data, exponent - 1) * self.grad;
        } else if (std.mem.eql(u8, self.op, "ReLU")) {
            self.prev.items[0].grad += if (self.data > 0) self.grad else 0;
        }
    }

    fn buildTopo(v: *Value, t: *std.ArrayList(*Value), vis: *std.AutoHashMap(*Value, void)) !void {
        if (!vis.contains(v)) {
            try vis.put(v, {});
            for (v.prev.items) |child| {
                try buildTopo(child, t, vis);
            }
            try t.append(v);
        }
    }

    pub fn format(self: Value, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("Value(data={d}, grad={d})", .{ self.data, self.grad });
    }
};
