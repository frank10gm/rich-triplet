// =============================================================================
// PyTorch `.bin` checkpoint reader -- ZIP container + pickle state dict
// =============================================================================
//
// Every other weight format this project reads is a flat binary with a
// self-describing header: GGUF, safetensors, and the internal cache files all
// say where each tensor is and stop there. `torch.save` instead writes a ZIP
// archive whose manifest is a Python pickle, so reading it means implementing
// two formats.
//
// Some published weights only exist in this format. SNAC's 24 kHz codec, which
// Orpheus decodes its audio tokens with, ships `pytorch_model.bin` and nothing
// else -- no safetensors mirror. Hence this file.
//
// ## Container layout
//
//   pytorch_model/data.pkl     the manifest: a pickled OrderedDict
//   pytorch_model/byteorder    b"little"
//   pytorch_model/data/0       raw tensor bytes, one entry per storage
//   pytorch_model/data/1
//   ...
//
// The archive directory prefix comes from whatever name `torch.save` was
// given, so it is discovered from the `data.pkl` entry rather than assumed.
// Entries are **stored**, never deflated -- tensor bytes do not compress, so
// PyTorch does not try. This reader rejects a compressed entry instead of
// pulling in an inflate implementation for a case that does not arise.
//
// ## Manifest layout
//
// The pickle is small and closed. It builds one `collections.OrderedDict`,
// fills it with `(name, tensor)` pairs, and attaches a `_metadata` dict that
// nothing here needs. Each tensor is a call to
//
//   torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride,
//                                   requires_grad, backward_hooks, metadata=None)
//
// where `storage` arrives as a persistent id -- `('storage', <StorageType>,
// <key>, <device>, <numel>)` -- naming the `data/<key>` archive entry.
//
// So the interpreter needs a value stack, a memo, three recognised globals
// (`collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, and a storage
// type), and the 24 opcodes those constructs emit. It is not a general
// unpickler and must never become one: a general unpickler executes arbitrary
// constructors named by the file, which is exactly the property that makes
// loading untrusted pickles unsafe. Anything outside the recognised set is an
// error, so a hostile or merely unusual file fails to parse rather than
// reaching for a constructor nobody vetted.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::autograd2::MatBf16;
use crate::gguf_loader::f16_to_f32;

// ---------------------------------------------------------------------------
// Byte reading
// ---------------------------------------------------------------------------

fn rd_u16(p: &[u8]) -> u16 {
    p[0] as u16 | ((p[1] as u16) << 8)
}

fn rd_u32(p: &[u8]) -> u32 {
    p[0] as u32 | ((p[1] as u32) << 8) | ((p[2] as u32) << 16) | ((p[3] as u32) << 24)
}

fn rd_u64(p: &[u8]) -> u64 {
    let mut v = 0u64;
    for i in (0..8).rev() {
        v = (v << 8) | p[i] as u64;
    }
    v
}

fn read_at(f: &mut File, offset: u64, len: usize) -> Result<Vec<u8>, String> {
    f.seek(SeekFrom::Start(offset))
        .map_err(|_| format!("torch_pickle: seek to {} failed", offset))?;
    let mut buf = vec![0u8; len];
    if len > 0 && f.read_exact(&mut buf).is_err() {
        return Err(format!("torch_pickle: short read of {} bytes at {}", len, offset));
    }
    Ok(buf)
}

fn file_size(f: &mut File) -> Result<u64, String> {
    f.seek(SeekFrom::End(0))
        .map_err(|_| "torch_pickle: seek to end failed".to_string())
}

// ---------------------------------------------------------------------------
// ZIP central directory
// ---------------------------------------------------------------------------

const EOCD_SIG: u32 = 0x06054b50; // "PK\5\6"
const EOCD64_SIG: u32 = 0x06064b50; // "PK\6\6"
const CENTRAL_SIG: u32 = 0x02014b50; // "PK\1\2"
const LOCAL_SIG: u32 = 0x04034b50; // "PK\3\4"
const METHOD_STORED: u16 = 0;

/// Where the central directory lives, and how many entries it has.
struct CentralDirInfo {
    offset: u64,
    count: u64,
}

/// Find the End Of Central Directory record by scanning backwards.
///
/// The EOCD sits at the end of the file but is followed by a variable-length
/// comment, so its position is not fixed and has to be searched for. 64 KiB is
/// the maximum a comment can be, so that bounds the scan.
fn find_central_dir(f: &mut File) -> Result<CentralDirInfo, String> {
    let total = file_size(f)?;
    const MAX_COMMENT: u64 = 65535 + 22;
    let window = if total < MAX_COMMENT { total } else { MAX_COMMENT };
    if window < 22 {
        return Err("torch_pickle: file too small to be a ZIP archive".to_string());
    }
    let tail = read_at(f, total - window, window as usize)?;

    let mut eocd: Option<usize> = None;
    for back in 22..=tail.len() {
        let pos = tail.len() - back;
        if rd_u32(&tail[pos..]) == EOCD_SIG {
            eocd = Some(pos);
            break;
        }
    }
    let Some(eocd) = eocd else {
        return Err("torch_pickle: no ZIP end-of-central-directory record found".to_string());
    };

    let e = &tail[eocd..];
    let mut info = CentralDirInfo { count: rd_u16(&e[10..]) as u64, offset: rd_u32(&e[16..]) as u64 };

    // A ZIP64 archive parks 0xffff / 0xffffffff in those fields and puts the
    // real values in a separate record. Checkpoints reach this size easily.
    if info.count == 0xffff || info.offset == 0xffffffff {
        let mut z64: Option<usize> = None;
        let mut pos = 0usize;
        while pos + 56 <= tail.len() {
            if rd_u32(&tail[pos..]) == EOCD64_SIG {
                z64 = Some(pos);
                break;
            }
            pos += 1;
        }
        let Some(z64) = z64 else {
            return Err("torch_pickle: ZIP64 archive without a ZIP64 EOCD record".to_string());
        };
        let z = &tail[z64..];
        info.count = rd_u64(&z[32..]);
        info.offset = rd_u64(&z[48..]);
    }
    Ok(info)
}

/// Resolve an entry's data offset by reading its local header.
fn local_data_offset(f: &mut File, header_offset: u64, name: &str) -> Result<u64, String> {
    let hdr = read_at(f, header_offset, 30)?;
    if rd_u32(&hdr) != LOCAL_SIG {
        return Err(format!("torch_pickle: bad local header signature for '{}'", name));
    }
    let method = rd_u16(&hdr[8..]);
    if method != METHOD_STORED {
        return Err(format!(
            "torch_pickle: entry '{}' uses compression method {}; only stored (0) is supported",
            name, method
        ));
    }
    let name_len = rd_u16(&hdr[26..]);
    let extra_len = rd_u16(&hdr[28..]);
    Ok(header_offset + 30 + name_len as u64 + extra_len as u64)
}

// ---------------------------------------------------------------------------
// ZIP container
// ---------------------------------------------------------------------------

/// One stored entry in the archive.
#[derive(Clone, Debug, Default)]
pub struct TorchZipEntry {
    pub name: String,
    /// Byte offset of the entry's data in the file, past its local header.
    pub data_offset: u64,
    /// Uncompressed length in bytes; equals the stored length.
    pub size: u64,
}

/// List every entry in a ZIP archive, resolving each one's true data offset.
///
/// The central directory records the offset of a *local header*, whose name and
/// extra fields can be longer than the central copy's, so the data offset has
/// to be computed by reading that header rather than by adding a fixed size.
///
/// Fails when any entry is compressed.
pub fn torch_zip_entries(path: &str) -> Result<Vec<TorchZipEntry>, String> {
    let mut f = File::open(path).map_err(|_| format!("torch_pickle: cannot open {}", path))?;

    let dir = find_central_dir(&mut f)?;

    let mut entries = Vec::with_capacity(dir.count as usize);

    let mut cursor = dir.offset;
    for i in 0..dir.count {
        let head = read_at(&mut f, cursor, 46)?;
        if rd_u32(&head) != CENTRAL_SIG {
            return Err(format!(
                "torch_pickle: bad central directory signature at entry {}",
                i
            ));
        }
        let method = rd_u16(&head[10..]);
        let mut comp_size = rd_u32(&head[20..]) as u64;
        let mut uncomp_size = rd_u32(&head[24..]) as u64;
        let name_len = rd_u16(&head[28..]);
        let extra_len = rd_u16(&head[30..]);
        let comment_len = rd_u16(&head[32..]);
        let mut local_offset = rd_u32(&head[42..]) as u64;

        let name_bytes = read_at(&mut f, cursor + 46, name_len as usize)?;
        let name = String::from_utf8_lossy(&name_bytes).into_owned();

        if method != METHOD_STORED {
            return Err(format!(
                "torch_pickle: entry '{}' uses compression method {}; only stored (0) is supported",
                name, method
            ));
        }

        // ZIP64 extra field (id 0x0001) carries the real sizes and offset when
        // the 32-bit fields are saturated. Values appear in a fixed order, but
        // only those that overflowed are present.
        if uncomp_size == 0xffffffff || comp_size == 0xffffffff || local_offset == 0xffffffff {
            let extra = read_at(&mut f, cursor + 46 + name_len as u64, extra_len as usize)?;
            let mut pos = 0usize;
            let mut found = false;
            while pos + 4 <= extra.len() {
                let id = rd_u16(&extra[pos..]);
                let len = rd_u16(&extra[pos + 2..]) as usize;
                if id == 0x0001 {
                    let mut at = pos + 4;
                    let end = at + len;
                    if uncomp_size == 0xffffffff && at + 8 <= end && at + 8 <= extra.len() {
                        uncomp_size = rd_u64(&extra[at..]);
                        at += 8;
                    }
                    if comp_size == 0xffffffff && at + 8 <= end && at + 8 <= extra.len() {
                        comp_size = rd_u64(&extra[at..]);
                        at += 8;
                    }
                    if local_offset == 0xffffffff && at + 8 <= end && at + 8 <= extra.len() {
                        local_offset = rd_u64(&extra[at..]);
                    }
                    found = true;
                    break;
                }
                pos += 4 + len;
            }
            if !found {
                return Err(format!(
                    "torch_pickle: entry '{}' needs a ZIP64 extra field but has none",
                    name
                ));
            }
        }

        if comp_size != uncomp_size {
            return Err(format!("torch_pickle: entry '{}' is not stored verbatim", name));
        }

        let data_off = local_data_offset(&mut f, local_offset, &name)?;

        entries.push(TorchZipEntry { name, data_offset: data_off, size: uncomp_size });

        cursor += 46 + name_len as u64 + extra_len as u64 + comment_len as u64;
    }

    Ok(entries)
}

// =============================================================================
// Pickle interpreter
// =============================================================================

/// Storage element types this reader understands, all widened to f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum StorageDtype {
    #[default]
    F32,
    F16,
    Bf16,
    F64,
}

fn dtype_from_global(name: &str) -> Option<StorageDtype> {
    match name {
        "torch FloatStorage" => Some(StorageDtype::F32),
        "torch HalfStorage" => Some(StorageDtype::F16),
        "torch BFloat16Storage" => Some(StorageDtype::Bf16),
        "torch DoubleStorage" => Some(StorageDtype::F64),
        _ => None,
    }
}

fn dtype_bytes(d: StorageDtype) -> usize {
    match d {
        StorageDtype::F32 => 4,
        StorageDtype::F16 | StorageDtype::Bf16 => 2,
        StorageDtype::F64 => 8,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum Kind {
    #[default]
    None,
    Mark,
    Int,
    Bool,
    Str,
    Tuple,
    Dict,
    Global,
    Storage,
    Tensor,
}

/// One value on the interpreter's stack.
///
/// A flat tagged struct rather than an enum with payloads: the set of shapes is
/// closed and small, and every field is cheap to leave empty.
#[derive(Clone, Debug, Default)]
struct Value {
    kind: Kind,
    integer: i64,
    boolean: bool,
    /// Str payload, or a global's "module name" pair joined by a space.
    text: String,
    items: Vec<Value>,
    entries: Vec<(Value, Value)>,

    // Storage
    dtype: StorageDtype,
    storage_key: String,
    storage_numel: usize,

    // Tensor
    shape: Vec<usize>,
    storage_offset: usize,
}

impl Value {
    fn of_kind(kind: Kind) -> Self {
        Value { kind, ..Default::default() }
    }
}

/// Interprets the closed subset of pickle that `torch.save` emits.
struct PickleMachine<'a> {
    b: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: Vec<Value>,
}

impl<'a> PickleMachine<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        PickleMachine { b: bytes, pos: 0, stack: Vec::new(), memo: Vec::new() }
    }

    fn byte(&mut self) -> Result<u8, String> {
        if self.pos >= self.b.len() {
            return Err("torch_pickle: pickle stream ended mid-opcode".to_string());
        }
        let v = self.b[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn u32(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.b.len() {
            return Err("torch_pickle: pickle stream ended mid-u32".to_string());
        }
        let v = rd_u32(&self.b[self.pos..]);
        self.pos += 4;
        Ok(v)
    }

    fn line(&mut self) -> Result<String, String> {
        let start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.b.len() {
            return Err("torch_pickle: unterminated pickle text field".to_string());
        }
        let s = String::from_utf8_lossy(&self.b[start..self.pos]).into_owned();
        self.pos += 1; // consume '\n'
        Ok(s)
    }

    fn pop(&mut self) -> Result<Value, String> {
        self.stack
            .pop()
            .ok_or_else(|| "torch_pickle: pickle stack underflow".to_string())
    }

    fn push(&mut self, v: Value) {
        self.stack.push(v);
    }

    fn memo_put(&mut self, index: usize) -> Result<(), String> {
        let Some(top) = self.stack.last() else {
            return Err("torch_pickle: memo put with an empty stack".to_string());
        };
        let top = top.clone();
        if index >= self.memo.len() {
            self.memo.resize(index + 1, Value::default());
        }
        self.memo[index] = top;
        Ok(())
    }

    fn memo_get(&mut self, index: usize) -> Result<(), String> {
        if index >= self.memo.len() {
            return Err(format!("torch_pickle: memo get of unset slot {}", index));
        }
        let v = self.memo[index].clone();
        self.stack.push(v);
        Ok(())
    }

    /// Pop everything above the most recent MARK, dropping the MARK itself.
    fn pop_to_mark(&mut self) -> Result<Vec<Value>, String> {
        let mut at = self.stack.len();
        while at > 0 && self.stack[at - 1].kind != Kind::Mark {
            at -= 1;
        }
        if at == 0 {
            return Err("torch_pickle: no MARK on the pickle stack".to_string());
        }
        let out = self.stack.split_off(at);
        self.stack.truncate(at - 1); // also drops the MARK
        Ok(out)
    }

    fn build_tensor(args: &[Value]) -> Result<Value, String> {
        // _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad,
        //                    backward_hooks, metadata=None)
        if args.len() < 4 {
            return Err(format!(
                "torch_pickle: _rebuild_tensor_v2 needs at least 4 arguments, got {}",
                args.len()
            ));
        }
        if args[0].kind != Kind::Storage {
            return Err("torch_pickle: _rebuild_tensor_v2 argument 0 is not a storage".to_string());
        }
        if args[1].kind != Kind::Int {
            return Err(
                "torch_pickle: _rebuild_tensor_v2 storage_offset is not an integer".to_string()
            );
        }
        if args[2].kind != Kind::Tuple || args[3].kind != Kind::Tuple {
            return Err("torch_pickle: _rebuild_tensor_v2 size/stride are not tuples".to_string());
        }
        if args[2].items.len() != args[3].items.len() {
            return Err("torch_pickle: _rebuild_tensor_v2 size and stride rank differ".to_string());
        }

        let mut t = args[0].clone(); // carry storage key, dtype, numel
        t.kind = Kind::Tensor;
        t.storage_offset = args[1].integer as usize;

        for d in &args[2].items {
            if d.kind != Kind::Int || d.integer < 0 {
                return Err(
                    "torch_pickle: _rebuild_tensor_v2 has a non-integer dimension".to_string()
                );
            }
            t.shape.push(d.integer as usize);
        }

        // Only contiguous tensors. A non-contiguous view would need its strides
        // honoured on read; silently ignoring them would scramble the weight.
        let mut expect: usize = 1;
        for i in (0..t.shape.len()).rev() {
            let s = &args[3].items[i];
            if s.kind != Kind::Int {
                return Err("torch_pickle: _rebuild_tensor_v2 has a non-integer stride".to_string());
            }
            // A dimension of length 1 can carry any stride without changing the
            // layout, so it is not worth rejecting.
            if t.shape[i] != 1 && s.integer as usize != expect {
                return Err(format!(
                    "torch_pickle: non-contiguous tensor (stride {} at dim {}, expected {})",
                    s.integer, i, expect
                ));
            }
            expect = expect.wrapping_mul(t.shape[i]);
        }

        Ok(t)
    }

    fn do_reduce(&mut self) -> Result<(), String> {
        let args = self.pop()?;
        let callable = self.pop()?;
        if callable.kind != Kind::Global {
            return Err("torch_pickle: REDUCE on a non-global callable".to_string());
        }
        if args.kind != Kind::Tuple {
            return Err("torch_pickle: REDUCE argument is not a tuple".to_string());
        }

        if callable.text == "collections OrderedDict" {
            self.push(Value::of_kind(Kind::Dict));
            return Ok(());
        }
        if callable.text == "torch._utils _rebuild_tensor_v2" {
            let t = Self::build_tensor(&args.items)?;
            self.push(t);
            return Ok(());
        }
        // Deliberately closed: a general unpickler would call whatever the file
        // names, which is the whole reason untrusted pickles are unsafe.
        Err(format!(
            "torch_pickle: refusing to call unrecognised global '{}'",
            callable.text
        ))
    }

    fn do_persid(&mut self) -> Result<(), String> {
        let pid = self.pop()?;
        if pid.kind != Kind::Tuple || pid.items.len() != 5 {
            return Err("torch_pickle: persistent id is not a 5-tuple".to_string());
        }
        if pid.items[0].kind != Kind::Str || pid.items[0].text != "storage" {
            return Err("torch_pickle: persistent id is not a storage reference".to_string());
        }
        if pid.items[1].kind != Kind::Global {
            return Err("torch_pickle: storage type is not a global".to_string());
        }
        let Some(dtype) = dtype_from_global(&pid.items[1].text) else {
            return Err(format!(
                "torch_pickle: unsupported storage type '{}' (only Float, Half, BFloat16 and Double are handled)",
                pid.items[1].text
            ));
        };
        if pid.items[2].kind != Kind::Str {
            return Err("torch_pickle: storage key is not a string".to_string());
        }
        if pid.items[4].kind != Kind::Int {
            return Err("torch_pickle: storage element count is not an integer".to_string());
        }

        let mut s = Value::of_kind(Kind::Storage);
        s.dtype = dtype;
        s.storage_key = pid.items[2].text.clone();
        s.storage_numel = pid.items[4].integer as usize;
        self.push(s);
        Ok(())
    }

    /// Run to STOP and return the final value.
    fn run(&mut self) -> Result<Value, String> {
        loop {
            let op = self.byte()?;
            match op {
                0x80 => {
                    // PROTO
                    let _proto = self.byte()?;
                }
                b'.' => {
                    // STOP
                    return self.pop();
                }
                b'(' => {
                    // MARK
                    self.push(Value::of_kind(Kind::Mark));
                }
                b'N' => {
                    // NONE
                    self.push(Value::default());
                }
                0x88 | 0x89 => {
                    // NEWTRUE / NEWFALSE
                    let mut v = Value::of_kind(Kind::Bool);
                    v.boolean = op == 0x88;
                    self.push(v);
                }
                b'K' => {
                    // BININT1
                    let n = self.byte()?;
                    let mut v = Value::of_kind(Kind::Int);
                    v.integer = n as i64;
                    self.push(v);
                }
                b'M' => {
                    // BININT2
                    let lo = self.byte()?;
                    let hi = self.byte()?;
                    let mut v = Value::of_kind(Kind::Int);
                    v.integer = lo as i64 | ((hi as i64) << 8);
                    self.push(v);
                }
                b'J' => {
                    // BININT (signed)
                    let n = self.u32()?;
                    let mut v = Value::of_kind(Kind::Int);
                    v.integer = n as i32 as i64;
                    self.push(v);
                }
                b'X' => {
                    // BINUNICODE
                    let len = self.u32()? as usize;
                    if self.pos + len > self.b.len() {
                        return Err(
                            "torch_pickle: BINUNICODE runs past the end of the stream".to_string()
                        );
                    }
                    let mut v = Value::of_kind(Kind::Str);
                    v.text = String::from_utf8_lossy(&self.b[self.pos..self.pos + len]).into_owned();
                    self.pos += len;
                    self.push(v);
                }
                b'c' => {
                    // GLOBAL
                    let module = self.line()?;
                    let name = self.line()?;
                    let mut v = Value::of_kind(Kind::Global);
                    v.text = format!("{} {}", module, name);
                    self.push(v);
                }
                b'q' => {
                    // BINPUT
                    let i = self.byte()?;
                    self.memo_put(i as usize)?;
                }
                b'r' => {
                    // LONG_BINPUT
                    let i = self.u32()?;
                    self.memo_put(i as usize)?;
                }
                b'h' => {
                    // BINGET
                    let i = self.byte()?;
                    self.memo_get(i as usize)?;
                }
                b'j' => {
                    // LONG_BINGET
                    let i = self.u32()?;
                    self.memo_get(i as usize)?;
                }
                b')' => {
                    // EMPTY_TUPLE
                    self.push(Value::of_kind(Kind::Tuple));
                }
                b'}' => {
                    // EMPTY_DICT
                    self.push(Value::of_kind(Kind::Dict));
                }
                b't' => {
                    // TUPLE
                    let items = self.pop_to_mark()?;
                    let mut v = Value::of_kind(Kind::Tuple);
                    v.items = items;
                    self.push(v);
                }
                0x85..=0x87 => {
                    // TUPLE1 / TUPLE2 / TUPLE3
                    let n = op as usize - 0x84;
                    if self.stack.len() < n {
                        return Err(format!("torch_pickle: TUPLE{} stack underflow", n));
                    }
                    let mut v = Value::of_kind(Kind::Tuple);
                    let at = self.stack.len() - n;
                    v.items = self.stack.split_off(at);
                    self.push(v);
                }
                b's' => {
                    // SETITEM
                    let value = self.pop()?;
                    let key = self.pop()?;
                    match self.stack.last_mut() {
                        Some(top) if top.kind == Kind::Dict => top.entries.push((key, value)),
                        _ => {
                            return Err(
                                "torch_pickle: SETITEM with no dict beneath it".to_string()
                            );
                        }
                    }
                }
                b'u' => {
                    // SETITEMS
                    let flat = self.pop_to_mark()?;
                    if flat.len() % 2 != 0 {
                        return Err(
                            "torch_pickle: SETITEMS with an odd number of values".to_string()
                        );
                    }
                    match self.stack.last_mut() {
                        Some(top) if top.kind == Kind::Dict => {
                            let mut it = flat.into_iter();
                            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                                top.entries.push((k, v));
                            }
                        }
                        _ => {
                            return Err(
                                "torch_pickle: SETITEMS with no dict beneath it".to_string()
                            );
                        }
                    }
                }
                b'R' => {
                    // REDUCE
                    self.do_reduce()?;
                }
                b'Q' => {
                    // BINPERSID
                    self.do_persid()?;
                }
                b'b' => {
                    // BUILD
                    // The state dict's `_metadata` arrives this way. Nothing here
                    // needs it; drop the state and leave the object.
                    let _state = self.pop()?;
                    if self.stack.is_empty() {
                        return Err("torch_pickle: BUILD with no object beneath it".to_string());
                    }
                }
                _ => {
                    return Err(format!(
                        "torch_pickle: unsupported pickle opcode 0x{:02x} at offset {}",
                        op,
                        self.pos - 1
                    ));
                }
            }
        }
    }
}

// =============================================================================
// Tensors
// =============================================================================

/// One tensor from a checkpoint, always converted to f32.
///
/// `data` is in row-major order, which is what a contiguous PyTorch tensor
/// already is -- non-contiguous tensors are rejected at load rather than
/// silently reordered.
#[derive(Clone, Debug, Default)]
pub struct TorchTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl TorchTensor {
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Product of the trailing dimensions after the first, or 1 when the shape
    /// has fewer than two dimensions.
    ///
    /// This is the group size `weight_norm_combine` normalizes over, and the
    /// row length when a `[out, in, k]` conv weight is viewed as `[out, in*k]`.
    pub fn inner_size(&self) -> usize {
        if self.shape.len() < 2 {
            return 1;
        }
        let mut n: usize = 1;
        for &d in &self.shape[1..] {
            n = n.wrapping_mul(d);
        }
        n
    }
}

/// A loaded state dict, keyed by parameter name.
#[derive(Clone, Debug, Default)]
pub struct TorchStateDict {
    pub tensors: BTreeMap<String, TorchTensor>,
}

fn fmt_shape(s: &[usize]) -> String {
    let parts: Vec<String> = s.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

impl TorchStateDict {
    /// Look up a parameter, or `None` when it is absent.
    pub fn find(&self, name: &str) -> Option<&TorchTensor> {
        self.tensors.get(name)
    }

    /// Look up a parameter of any shape, erroring when absent.
    pub fn require(&self, name: &str) -> Result<&TorchTensor, String> {
        self.find(name)
            .ok_or_else(|| format!("torch_pickle: checkpoint has no tensor '{}'", name))
    }

    /// Look up a parameter, requiring an exact shape. Returns an error naming
    /// the parameter and both shapes on a mismatch -- a wrong shape in a codec
    /// decoder produces noise rather than a crash, so it is worth failing
    /// loudly at load.
    pub fn require_shape(&self, name: &str, shape: &[usize]) -> Result<&TorchTensor, String> {
        let t = self.require(name)?;
        if t.shape != shape {
            return Err(format!(
                "torch_pickle: tensor '{}' has shape {}, expected {}",
                name,
                fmt_shape(&t.shape),
                fmt_shape(shape)
            ));
        }
        Ok(t)
    }
}

/// Read a `torch.save`d state dict.
///
/// Handles f32, f16, bf16 and f64 storages, converting every one to f32.
/// Rejects: compressed archive entries, integer and boolean storages,
/// non-contiguous tensors, and any pickle opcode or global outside the closed
/// set the format needs.
pub fn load_torch_state_dict(path: &str) -> Result<TorchStateDict, String> {
    let entries = torch_zip_entries(path)?;

    // The archive prefix comes from the name torch.save was given, so find it
    // rather than assuming "pytorch_model/".
    let Some(manifest) = entries.iter().find(|e| e.name.as_bytes().ends_with(b"data.pkl")) else {
        return Err("torch_pickle: archive has no data.pkl manifest".to_string());
    };
    let prefix = manifest.name[..manifest.name.len() - 8].to_string();

    let mut by_name: BTreeMap<&str, &TorchZipEntry> = BTreeMap::new();
    for e in &entries {
        by_name.entry(e.name.as_str()).or_insert(e);
    }

    let mut f = File::open(path).map_err(|_| format!("torch_pickle: cannot reopen {}", path))?;

    let pkl = read_at(&mut f, manifest.data_offset, manifest.size as usize)?;

    let mut machine = PickleMachine::new(&pkl);
    let root = machine.run()?;
    if root.kind != Kind::Dict {
        return Err("torch_pickle: manifest root is not a dict".to_string());
    }

    // One storage can back several tensors, so cache the decoded bytes.
    let mut storage_cache: BTreeMap<String, Vec<f32>> = BTreeMap::new();

    let mut out = TorchStateDict::default();
    for (key, value) in &root.entries {
        if key.kind != Kind::Str || value.kind != Kind::Tensor {
            continue; // `_metadata` and friends
        }

        let mut want: usize = 1;
        for &d in &value.shape {
            want = want.wrapping_mul(d);
        }

        if !storage_cache.contains_key(&value.storage_key) {
            let entry_name = format!("{}data/{}", prefix, value.storage_key);
            let Some(entry) = by_name.get(entry_name.as_str()) else {
                return Err(format!(
                    "torch_pickle: tensor '{}' references missing entry '{}'",
                    key.text, entry_name
                ));
            };
            let elem = dtype_bytes(value.dtype);
            let need = (value.storage_numel as u64).wrapping_mul(elem as u64);
            if entry.size < need {
                return Err(format!(
                    "torch_pickle: entry '{}' holds {} bytes, needs {}",
                    entry_name, entry.size, need
                ));
            }
            let raw = read_at(&mut f, entry.data_offset, need as usize)?;

            let mut floats = vec![0.0f32; value.storage_numel];
            for (i, dst) in floats.iter_mut().enumerate() {
                let p = &raw[i * elem..];
                *dst = match value.dtype {
                    StorageDtype::F32 => f32::from_bits(rd_u32(p)),
                    StorageDtype::F16 => f16_to_f32(rd_u16(p)),
                    StorageDtype::Bf16 => MatBf16::bf16_to_f32(rd_u16(p)),
                    StorageDtype::F64 => f64::from_bits(rd_u64(p)) as f32,
                };
            }
            storage_cache.insert(value.storage_key.clone(), floats);
        }

        let storage = &storage_cache[&value.storage_key];
        if value.storage_offset.wrapping_add(want) > storage.len() {
            return Err(format!(
                "torch_pickle: tensor '{}' runs past its storage (offset {} + {} > {})",
                key.text,
                value.storage_offset,
                want,
                storage.len()
            ));
        }

        let tensor = TorchTensor {
            shape: value.shape.clone(),
            data: storage[value.storage_offset..value.storage_offset + want].to_vec(),
        };
        out.tensors.entry(key.text.clone()).or_insert(tensor);
    }

    if out.tensors.is_empty() {
        return Err("torch_pickle: manifest contained no tensors".to_string());
    }
    Ok(out)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
pub(crate) mod testing {
    // Builds `torch.save`-shaped archives in memory, so the reader can be tested
    // without a Python interpreter in the loop.
    //
    // Only what the reader has to cope with is modelled: stored ZIP entries, a
    // pickle manifest of `_rebuild_tensor_v2` calls, and the knobs needed to
    // produce the malformed cases (compression, bad strides, unknown globals,
    // unsupported opcodes).

    // -----------------------------------------------------------------------
    // Pickle emission
    // -----------------------------------------------------------------------

    #[derive(Default)]
    pub struct PickleBuilder {
        out: Vec<u8>,
    }

    impl PickleBuilder {
        pub fn proto(&mut self, version: u8) {
            self.put(0x80);
            self.put(version);
        }
        pub fn stop(&mut self) {
            self.put(b'.');
        }
        pub fn mark(&mut self) {
            self.put(b'(');
        }
        pub fn empty_tuple(&mut self) {
            self.put(b')');
        }
        pub fn empty_dict(&mut self) {
            self.put(b'}');
        }
        pub fn tuple(&mut self) {
            self.put(b't');
        }
        pub fn tuple1(&mut self) {
            self.put(0x85);
        }
        pub fn tuple2(&mut self) {
            self.put(0x86);
        }
        pub fn tuple3(&mut self) {
            self.put(0x87);
        }
        pub fn setitems(&mut self) {
            self.put(b'u');
        }
        pub fn setitem(&mut self) {
            self.put(b's');
        }
        pub fn reduce(&mut self) {
            self.put(b'R');
        }
        pub fn persid(&mut self) {
            self.put(b'Q');
        }
        pub fn build(&mut self) {
            self.put(b'b');
        }
        pub fn newfalse(&mut self) {
            self.put(0x89);
        }
        pub fn newtrue(&mut self) {
            self.put(0x88);
        }
        pub fn none(&mut self) {
            self.put(b'N');
        }

        pub fn global(&mut self, module: &str, name: &str) {
            self.put(b'c');
            self.raw(module);
            self.put(b'\n');
            self.raw(name);
            self.put(b'\n');
        }

        pub fn unicode(&mut self, s: &str) {
            self.put(b'X');
            self.u32(s.len() as u32);
            self.raw(s);
        }

        /// Smallest integer encoding that fits, like the real pickler.
        pub fn integer(&mut self, v: i64) {
            if (0..256).contains(&v) {
                self.put(b'K');
                self.put(v as u8);
            } else if (0..65536).contains(&v) {
                self.put(b'M');
                self.put((v & 0xff) as u8);
                self.put(((v >> 8) & 0xff) as u8);
            } else {
                self.put(b'J');
                self.u32(v as i32 as u32);
            }
        }

        pub fn binput(&mut self, index: u32) {
            if index < 256 {
                self.put(b'q');
                self.put(index as u8);
            } else {
                self.put(b'r');
                self.u32(index);
            }
        }

        pub fn binget(&mut self, index: u32) {
            if index < 256 {
                self.put(b'h');
                self.put(index as u8);
            } else {
                self.put(b'j');
                self.u32(index);
            }
        }

        /// Raw opcode escape hatch, for the unsupported-opcode path.
        pub fn opcode(&mut self, op: u8) {
            self.put(op);
        }

        pub fn bytes(&self) -> &[u8] {
            &self.out
        }

        fn put(&mut self, b: u8) {
            self.out.push(b);
        }
        fn raw(&mut self, s: &str) {
            self.out.extend_from_slice(s.as_bytes());
        }
        fn u32(&mut self, v: u32) {
            for i in 0..4 {
                self.put(((v >> (8 * i)) & 0xff) as u8);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tensor description
    // -----------------------------------------------------------------------

    #[derive(Clone)]
    pub struct FakeTensor {
        pub name: String,
        pub storage_key: String,
        pub shape: Vec<usize>,
        /// Number of elements in the backing storage entry.
        pub storage_numel: usize,
        /// Element offset into that storage.
        pub storage_offset: usize,
        /// Override the emitted strides; empty means emit contiguous ones.
        pub strides: Vec<i64>,
        /// "FloatStorage", "HalfStorage", "BFloat16Storage", "DoubleStorage".
        pub storage_type: String,
        /// Use TUPLE1/2/3 rather than MARK ... TUPLE for the size and stride.
        pub short_tuples: bool,
    }

    impl Default for FakeTensor {
        fn default() -> Self {
            FakeTensor {
                name: String::new(),
                storage_key: String::new(),
                shape: Vec::new(),
                storage_numel: 0,
                storage_offset: 0,
                strides: Vec::new(),
                storage_type: "FloatStorage".to_string(),
                short_tuples: false,
            }
        }
    }

    /// Emit a manifest for a list of tensors.
    ///
    /// `rebuild_global` is the name called for each tensor; overriding it exercises
    /// the reader's refusal to invoke globals it does not recognise. When
    /// `with_metadata` is set, a `_metadata` entry is attached through BUILD, the
    /// way a real state dict carries it.
    pub fn build_manifest(tensors: &[FakeTensor], with_metadata: bool, rebuild_global: &str) -> Vec<u8> {
        let mut p = PickleBuilder::default();
        p.proto(2);
        p.global("collections", "OrderedDict");
        p.binput(0);
        p.empty_tuple();
        p.reduce();
        p.binput(1);
        p.mark();

        for t in tensors {
            p.unicode(&t.name);
            p.global("torch._utils", rebuild_global);

            p.mark(); // arguments to _rebuild_tensor_v2

            // storage: BINPERSID over ('storage', <Type>, key, 'cpu', numel)
            p.mark();
            p.unicode("storage");
            p.global("torch", &t.storage_type);
            p.unicode(&t.storage_key);
            p.unicode("cpu");
            p.integer(t.storage_numel as i64);
            p.tuple();
            p.persid();

            p.integer(t.storage_offset as i64);

            let emit_tuple = |p: &mut PickleBuilder, values: &[i64]| {
                if t.short_tuples && !values.is_empty() && values.len() <= 3 {
                    for &v in values {
                        p.integer(v);
                    }
                    match values.len() {
                        1 => p.tuple1(),
                        2 => p.tuple2(),
                        _ => p.tuple3(),
                    }
                } else {
                    p.mark();
                    for &v in values {
                        p.integer(v);
                    }
                    p.tuple();
                }
            };

            let size: Vec<i64> = t.shape.iter().map(|&d| d as i64).collect();
            emit_tuple(&mut p, &size);

            let mut stride = t.strides.clone();
            if stride.is_empty() {
                stride = vec![1; t.shape.len()];
                for i in (1..t.shape.len()).rev() {
                    stride[i - 1] = stride[i] * t.shape[i] as i64;
                }
            }
            emit_tuple(&mut p, &stride);

            p.newfalse(); // requires_grad
            p.empty_dict(); // backward_hooks
            p.tuple(); // close the argument tuple
            p.reduce(); // -> tensor
        }

        p.setitems();

        if with_metadata {
            // A real state dict attaches `{'_metadata': {...}}` via BUILD. Nothing
            // in the reader needs it, so it must be skipped rather than parsed.
            p.empty_dict();
            p.unicode("_metadata");
            p.empty_dict();
            p.setitem();
            p.build();
        }

        p.stop();
        p.bytes().to_vec()
    }

    // -----------------------------------------------------------------------
    // ZIP emission
    // -----------------------------------------------------------------------

    #[derive(Clone, Default)]
    pub struct ZipMember {
        pub name: String,
        pub data: Vec<u8>,
        /// Non-zero claims a compression method the reader must reject.
        pub method: u16,
    }

    impl ZipMember {
        pub fn new(name: &str, data: Vec<u8>) -> Self {
            ZipMember { name: name.to_string(), data, method: 0 }
        }
    }

    /// Assemble a ZIP archive from stored members.
    pub fn build_zip(members: &[ZipMember]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        fn u16(out: &mut Vec<u8>, v: u16) {
            out.push((v & 0xff) as u8);
            out.push(((v >> 8) & 0xff) as u8);
        }
        fn u32(out: &mut Vec<u8>, v: u32) {
            for i in 0..4 {
                out.push(((v >> (8 * i)) & 0xff) as u8);
            }
        }

        let mut local_offsets = Vec::new();
        for m in members {
            local_offsets.push(out.len() as u32);
            u32(&mut out, 0x04034b50); // local header signature
            u16(&mut out, 20); // version needed
            u16(&mut out, 0); // flags
            u16(&mut out, m.method);
            u16(&mut out, 0); // mod time
            u16(&mut out, 0); // mod date
            u32(&mut out, 0); // crc32 -- not verified by the reader
            u32(&mut out, m.data.len() as u32);
            u32(&mut out, m.data.len() as u32);
            u16(&mut out, m.name.len() as u16);
            u16(&mut out, 0); // extra length
            out.extend_from_slice(m.name.as_bytes());
            out.extend_from_slice(&m.data);
        }

        let cd_start = out.len() as u32;
        for (i, m) in members.iter().enumerate() {
            u32(&mut out, 0x02014b50); // central directory signature
            u16(&mut out, 20); // version made by
            u16(&mut out, 20); // version needed
            u16(&mut out, 0); // flags
            u16(&mut out, m.method);
            u16(&mut out, 0); // mod time
            u16(&mut out, 0); // mod date
            u32(&mut out, 0); // crc32
            u32(&mut out, m.data.len() as u32);
            u32(&mut out, m.data.len() as u32);
            u16(&mut out, m.name.len() as u16);
            u16(&mut out, 0); // extra length
            u16(&mut out, 0); // comment length
            u16(&mut out, 0); // disk number start
            u16(&mut out, 0); // internal attributes
            u32(&mut out, 0); // external attributes
            u32(&mut out, local_offsets[i]);
            out.extend_from_slice(m.name.as_bytes());
        }
        let cd_size = out.len() as u32 - cd_start;

        u32(&mut out, 0x06054b50); // end of central directory
        u16(&mut out, 0); // this disk
        u16(&mut out, 0); // disk with the central directory
        u16(&mut out, members.len() as u16);
        u16(&mut out, members.len() as u16);
        u32(&mut out, cd_size);
        u32(&mut out, cd_start);
        u16(&mut out, 0); // comment length
        out
    }

    /// Raw little-endian f32 bytes.
    pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
    }

    /// Raw little-endian u16 bytes, for f16 and bf16 storages.
    pub fn u16_bytes(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Raw little-endian f64 bytes.
    pub fn f64_bytes(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
    }

    /// Write bytes to `path`, returning false on any I/O failure.
    pub fn write_file(path: &str, bytes: &[u8]) -> bool {
        std::fs::write(path, bytes).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    /// A checkpoint written to a temp path, removed when the test finishes.
    struct TempCheckpoint {
        path: String,
    }

    impl TempCheckpoint {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("rt_torch_pickle_{}.bin", tag))
                .to_string_lossy()
                .into_owned();
            TempCheckpoint { path }
        }

        fn path(&self) -> &str {
            &self.path
        }

        fn write(&self, members: &[ZipMember]) -> bool {
            write_file(&self.path, &build_zip(members))
        }
    }

    impl Drop for TempCheckpoint {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn manifest(tensors: &[FakeTensor]) -> Vec<u8> {
        build_manifest(tensors, true, "_rebuild_tensor_v2")
    }

    fn fake(name: &str, key: &str, shape: &[usize], numel: usize) -> FakeTensor {
        FakeTensor {
            name: name.to_string(),
            storage_key: key.to_string(),
            shape: shape.to_vec(),
            storage_numel: numel,
            ..Default::default()
        }
    }

    /// A two-tensor checkpoint: a [2,3] matrix and a [4] vector.
    fn simple_members() -> Vec<ZipMember> {
        let tensors = vec![fake("layer.weight", "0", &[2, 3], 6), fake("layer.bias", "1", &[4], 4)];
        vec![
            ZipMember::new("archive/data.pkl", manifest(&tensors)),
            ZipMember::new("archive/data/0", f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            ZipMember::new("archive/data/1", f32_bytes(&[-1.0, -2.0, -3.0, -4.0])),
        ]
    }

    // =========================================================================
    // ZIP container
    // =========================================================================

    #[test]
    fn torch_zip_entries_lists_names_sizes_and_data_offsets() {
        let ckpt = TempCheckpoint::new("zip_list");
        let members = simple_members();
        assert!(ckpt.write(&members));

        let entries = torch_zip_entries(ckpt.path()).expect("entries");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, "archive/data.pkl");
        assert_eq!(entries[1].name, "archive/data/0");
        assert_eq!(entries[1].size, 24);
        assert_eq!(entries[2].size, 16);

        // The data offset must point at the payload, not the local header, so the
        // first four bytes of entry 1 are the f32 1.0 pattern.
        let bytes = std::fs::read(ckpt.path()).expect("read");
        let off = entries[1].data_offset as usize;
        let first = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        assert!(approx(first, 1.0));
    }

    #[test]
    fn torch_zip_entries_rejects_a_compressed_entry() {
        let ckpt = TempCheckpoint::new("compressed");
        let mut members = simple_members();
        members[1].method = 8; // deflate
        assert!(ckpt.write(&members));

        let entries = torch_zip_entries(ckpt.path());
        assert!(entries.unwrap_err().contains("compression method 8"));
    }

    #[test]
    fn torch_zip_entries_fails_cleanly_on_a_non_archive() {
        let ckpt = TempCheckpoint::new("garbage");
        assert!(write_file(ckpt.path(), &[0x41u8; 512]));
        assert!(torch_zip_entries(ckpt.path()).is_err());
    }

    #[test]
    fn torch_zip_entries_fails_cleanly_on_a_missing_file() {
        let entries = torch_zip_entries("/tmp/rt_definitely_not_here.bin");
        assert!(entries.unwrap_err().contains("cannot open"));
    }

    // =========================================================================
    // State dict
    // =========================================================================

    #[test]
    fn load_torch_state_dict_reads_shapes_and_values() {
        let ckpt = TempCheckpoint::new("simple");
        assert!(ckpt.write(&simple_members()));

        let sd = load_torch_state_dict(ckpt.path()).expect("load");
        assert_eq!(sd.tensors.len(), 2);

        let w = sd.find("layer.weight").expect("weight");
        assert_eq!(w.shape, vec![2, 3]);
        assert_eq!(w.numel(), 6);
        for i in 0..6 {
            assert!(approx(w.data[i], (i + 1) as f32));
        }

        let b = sd.find("layer.bias").expect("bias");
        assert_eq!(b.shape, vec![4]);
        assert!(approx(b.data[3], -4.0));

        assert!(sd.find("layer.absent").is_none());
    }

    #[test]
    fn load_torch_state_dict_discovers_the_archive_prefix() {
        // The directory name comes from whatever torch.save was called with, so a
        // hardcoded "pytorch_model/" would fail here.
        let ckpt = TempCheckpoint::new("prefix");
        let tensors = vec![fake("w", "0", &[2], 2)];
        assert!(ckpt.write(&[
            ZipMember::new("some_other_name/data.pkl", manifest(&tensors)),
            ZipMember::new("some_other_name/data/0", f32_bytes(&[7.0, 8.0])),
        ]));

        let sd = load_torch_state_dict(ckpt.path()).expect("load");
        assert!(approx(sd.find("w").unwrap().data[1], 8.0));
    }

    #[test]
    fn load_torch_state_dict_shares_one_storage_between_tensors() {
        // Two views into the same storage at different offsets -- how weight-tied
        // or sliced parameters get saved.
        let ckpt = TempCheckpoint::new("shared");
        let tensors = vec![
            FakeTensor { storage_offset: 0, ..fake("first", "0", &[3], 6) },
            FakeTensor { storage_offset: 3, ..fake("second", "0", &[3], 6) },
        ];
        assert!(ckpt.write(&[
            ZipMember::new("a/data.pkl", manifest(&tensors)),
            ZipMember::new("a/data/0", f32_bytes(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
        ]));

        let sd = load_torch_state_dict(ckpt.path()).expect("load");
        assert!(approx(sd.find("first").unwrap().data[0], 10.0));
        assert!(approx(sd.find("second").unwrap().data[0], 40.0));
        assert!(approx(sd.find("second").unwrap().data[2], 60.0));
    }

    #[test]
    fn load_torch_state_dict_accepts_tuple1_2_3_shapes() {
        // The real pickler uses the short tuple opcodes for ranks 1-3, which is
        // every conv weight in a codec decoder.
        let ckpt = TempCheckpoint::new("short_tuples");
        let tensors = vec![
            FakeTensor { short_tuples: true, ..fake("rank1", "0", &[2], 2) },
            FakeTensor { short_tuples: true, ..fake("rank3", "1", &[2, 2, 2], 8) },
        ];
        assert!(ckpt.write(&[
            ZipMember::new("a/data.pkl", manifest(&tensors)),
            ZipMember::new("a/data/0", f32_bytes(&[1.0, 2.0])),
            ZipMember::new("a/data/1", f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
        ]));

        let sd = load_torch_state_dict(ckpt.path()).expect("load");
        assert_eq!(sd.find("rank1").unwrap().shape, vec![2]);
        assert_eq!(sd.find("rank3").unwrap().shape, vec![2, 2, 2]);
        assert!(approx(sd.find("rank3").unwrap().data[7], 8.0));
    }

    #[test]
    fn load_torch_state_dict_skips_the_metadata_entry() {
        let ckpt = TempCheckpoint::new("metadata");
        assert!(ckpt.write(&simple_members()));
        let sd = load_torch_state_dict(ckpt.path()).expect("load");
        // build_manifest attaches `_metadata` by default; only real tensors land.
        assert_eq!(sd.tensors.len(), 2);
        assert!(sd.find("_metadata").is_none());
    }

    #[test]
    fn load_torch_state_dict_works_without_a_metadata_entry() {
        let ckpt = TempCheckpoint::new("no_metadata");
        let tensors = vec![fake("w", "0", &[2], 2)];
        assert!(ckpt.write(&[
            ZipMember::new("a/data.pkl", build_manifest(&tensors, false, "_rebuild_tensor_v2")),
            ZipMember::new("a/data/0", f32_bytes(&[1.0, 2.0])),
        ]));
        let sd = load_torch_state_dict(ckpt.path()).expect("load");
        assert_eq!(sd.tensors.len(), 1);
    }

    // =========================================================================
    // Storage dtypes
    // =========================================================================

    #[test]
    fn load_torch_state_dict_widens_f16_bf16_and_f64_to_f32() {
        let ckpt = TempCheckpoint::new("dtypes");
        let tensors = vec![
            FakeTensor { storage_type: "HalfStorage".to_string(), ..fake("half", "0", &[2], 2) },
            FakeTensor { storage_type: "BFloat16Storage".to_string(), ..fake("bfloat", "1", &[2], 2) },
            FakeTensor { storage_type: "DoubleStorage".to_string(), ..fake("double", "2", &[2], 2) },
        ];

        // 1.0 and -2.0 in each encoding.
        let halves = [0x3c00u16, 0xc000];
        let bfloats = [0x3f80u16, 0xc000];

        assert!(ckpt.write(&[
            ZipMember::new("a/data.pkl", manifest(&tensors)),
            ZipMember::new("a/data/0", u16_bytes(&halves)),
            ZipMember::new("a/data/1", u16_bytes(&bfloats)),
            ZipMember::new("a/data/2", f64_bytes(&[1.0, -2.0])),
        ]));

        let sd = load_torch_state_dict(ckpt.path()).expect("load");
        for name in ["half", "bfloat", "double"] {
            let t = sd.find(name).expect(name);
            assert!(approx(t.data[0], 1.0));
            assert!(approx(t.data[1], -2.0));
        }
    }

    #[test]
    fn load_torch_state_dict_rejects_an_integer_storage() {
        let ckpt = TempCheckpoint::new("int_storage");
        let tensors =
            vec![FakeTensor { storage_type: "LongStorage".to_string(), ..fake("counts", "0", &[2], 2) }];
        assert!(ckpt.write(&[
            ZipMember::new("a/data.pkl", manifest(&tensors)),
            ZipMember::new("a/data/0", vec![0u8; 16]),
        ]));

        let sd = load_torch_state_dict(ckpt.path());
        assert!(sd.unwrap_err().contains("unsupported storage type"));
    }

    // =========================================================================
    // Rejections
    // =========================================================================

    #[test]
    fn load_torch_state_dict_rejects_a_non_contiguous_tensor() {
        // A transposed view: shape [2,3] with strides [1,2]. Ignoring the strides
        // would scramble the weight, so this has to fail rather than load.
        let ckpt = TempCheckpoint::new("strided");
        let tensors = vec![FakeTensor { strides: vec![1, 2], ..fake("w", "0", &[2, 3], 6) }];
        assert!(ckpt.write(&[
            ZipMember::new("a/data.pkl", manifest(&tensors)),
            ZipMember::new("a/data/0", f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ]));

        let sd = load_torch_state_dict(ckpt.path());
        assert!(sd.unwrap_err().contains("non-contiguous"));
    }

    #[test]
    fn load_torch_state_dict_tolerates_any_stride_on_a_length_1_dim() {
        // A dimension of length 1 can carry any stride without changing the
        // layout, which is how [C, 1, K] depthwise conv weights are often saved.
        let ckpt = TempCheckpoint::new("unit_dim");
        let tensors = vec![FakeTensor { strides: vec![3, 999, 1], ..fake("w", "0", &[2, 1, 3], 6) }];
        assert!(ckpt.write(&[
            ZipMember::new("a/data.pkl", manifest(&tensors)),
            ZipMember::new("a/data/0", f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ]));

        let sd = load_torch_state_dict(ckpt.path()).expect("load");
        assert_eq!(sd.find("w").unwrap().shape, vec![2, 1, 3]);
    }

    #[test]
    fn load_torch_state_dict_refuses_an_unrecognised_global() {
        // The security property: a general unpickler calls whatever constructor the
        // file names. This reader knows three globals and rejects everything else,
        // so a hostile file fails to parse instead of reaching for code nobody
        // vetted.
        let ckpt = TempCheckpoint::new("bad_global");
        let tensors = vec![fake("w", "0", &[2], 2)];
        assert!(ckpt.write(&[
            ZipMember::new("a/data.pkl", build_manifest(&tensors, true, "definitely_not_rebuild_tensor")),
            ZipMember::new("a/data/0", f32_bytes(&[1.0, 2.0])),
        ]));

        let sd = load_torch_state_dict(ckpt.path());
        assert!(sd.unwrap_err().contains("refusing to call unrecognised global"));
    }

    #[test]
    fn load_torch_state_dict_rejects_an_unsupported_opcode() {
        let ckpt = TempCheckpoint::new("bad_opcode");
        let mut p = PickleBuilder::default();
        p.proto(2);
        p.opcode(b'I'); // INT -- text-mode integer, outside the supported set
        let pkl = p.bytes().to_vec();
        assert!(ckpt.write(&[ZipMember::new("a/data.pkl", pkl)]));

        let sd = load_torch_state_dict(ckpt.path());
        assert!(sd.unwrap_err().contains("unsupported pickle opcode 0x49"));
    }

    #[test]
    fn load_torch_state_dict_reports_a_missing_storage_entry() {
        let ckpt = TempCheckpoint::new("missing_storage");
        let tensors = vec![fake("w", "7", &[2], 2)];
        assert!(ckpt.write(&[
            ZipMember::new("a/data.pkl", manifest(&tensors)),
            ZipMember::new("a/data/0", f32_bytes(&[1.0, 2.0])),
        ]));

        let sd = load_torch_state_dict(ckpt.path());
        assert!(sd.unwrap_err().contains("missing entry"));
    }

    #[test]
    fn load_torch_state_dict_reports_a_storage_that_is_too_small() {
        let ckpt = TempCheckpoint::new("short_storage");
        let tensors = vec![fake("w", "0", &[8], 8)];
        assert!(ckpt.write(&[
            ZipMember::new("a/data.pkl", manifest(&tensors)),
            ZipMember::new("a/data/0", f32_bytes(&[1.0, 2.0])),
        ]));

        let sd = load_torch_state_dict(ckpt.path());
        assert!(sd.unwrap_err().contains("needs"));
    }

    #[test]
    fn load_torch_state_dict_reports_a_manifest_with_no_tensors() {
        let ckpt = TempCheckpoint::new("empty");
        assert!(ckpt.write(&[ZipMember::new("a/data.pkl", build_manifest(&[], true, "_rebuild_tensor_v2"))]));
        let sd = load_torch_state_dict(ckpt.path());
        assert!(sd.unwrap_err().contains("no tensors"));
    }

    #[test]
    fn load_torch_state_dict_reports_an_archive_with_no_manifest() {
        let ckpt = TempCheckpoint::new("no_manifest");
        assert!(ckpt.write(&[ZipMember::new("a/data/0", f32_bytes(&[1.0]))]));
        let sd = load_torch_state_dict(ckpt.path());
        assert!(sd.unwrap_err().contains("no data.pkl"));
    }

    // =========================================================================
    // Lookup helpers
    // =========================================================================

    #[test]
    fn torch_state_dict_require_names_both_shapes_on_a_mismatch() {
        let ckpt = TempCheckpoint::new("require");
        assert!(ckpt.write(&simple_members()));
        let sd = load_torch_state_dict(ckpt.path()).expect("load");

        assert!(sd.require_shape("layer.weight", &[2, 3]).is_ok());

        let bad = sd.require_shape("layer.weight", &[3, 2]).unwrap_err();
        assert!(bad.contains("[2, 3]"));
        assert!(bad.contains("[3, 2]"));

        let absent = sd.require("nope").unwrap_err();
        assert!(absent.contains("no tensor 'nope'"));
    }

    #[test]
    fn torch_tensor_inner_size_is_the_product_past_the_first_axis() {
        let mut t = TorchTensor::default();
        t.shape = vec![4];
        assert_eq!(t.inner_size(), 1);
        t.shape = vec![4, 5];
        assert_eq!(t.inner_size(), 5);
        t.shape = vec![1024, 512, 16];
        assert_eq!(t.inner_size(), 512 * 16);
    }

    // =========================================================================
    // The real checkpoint
    // =========================================================================

    #[test]
    fn load_torch_state_dict_reads_the_snac_24khz_checkpoint() {
        // Skipped unless the weights have been fetched. Shapes are the ones the
        // decoder depends on, including the transposed convolution whose
        // weight-norm magnitude is indexed by *input* channel.
        let path = "models/snac_24khz.bin";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: models/snac_24khz.bin not present");
            return;
        }

        let sd = load_torch_state_dict(path).expect("load");
        assert_eq!(sd.tensors.len(), 269);

        assert!(sd.require_shape("decoder.model.0.parametrizations.weight.original1", &[768, 1, 7]).is_ok());
        assert!(sd.require_shape("decoder.model.1.parametrizations.weight.original1", &[1024, 768, 1]).is_ok());
        assert!(sd
            .require_shape("decoder.model.2.block.1.parametrizations.weight.original1", &[1024, 512, 16])
            .is_ok());
        assert!(sd
            .require_shape("decoder.model.2.block.1.parametrizations.weight.original0", &[1024, 1, 1])
            .is_ok());
        assert!(sd.require_shape("decoder.model.2.block.1.bias", &[512]).is_ok());
        assert!(sd.require_shape("decoder.model.2.block.0.alpha", &[1, 1024, 1]).is_ok());
        assert!(sd.require_shape("decoder.model.7.parametrizations.weight.original1", &[1, 64, 7]).is_ok());
        assert!(sd.require_shape("quantizer.quantizers.0.codebook.weight", &[4096, 8]).is_ok());
        assert!(sd
            .require_shape("quantizer.quantizers.2.out_proj.parametrizations.weight.original1", &[768, 8, 1])
            .is_ok());

        // The NoiseBlock convolution has no bias, unlike every other conv here.
        assert!(sd.find("decoder.model.2.block.2.linear.bias").is_none());
        assert!(sd
            .require_shape("decoder.model.2.block.2.linear.parametrizations.weight.original1", &[512, 512, 1])
            .is_ok());

        // Weights should be finite and not all zero.
        let alpha = sd.find("decoder.model.6.alpha").expect("alpha");
        assert_eq!(alpha.numel(), 64);
        let mut sum_abs = 0.0f32;
        for &v in &alpha.data {
            assert!(v.is_finite());
            sum_abs += v.abs();
        }
        assert!(sum_abs > 0.0);
    }
}
