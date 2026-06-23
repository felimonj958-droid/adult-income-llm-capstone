# Compatibility shim: some models serialized with older scikit-learn versions
# expect internal symbols that may not exist in newer sklearn packages. Add
# a minimal stand-in for `_RemainderColsList` to support unpickling such models
# during tests.
try:
	import sklearn.compose._column_transformer as _ct  # type: ignore

	if not hasattr(_ct, "_RemainderColsList"):
		class _RemainderColsList(list):
			"""Minimal stand-in used only for unpickling compatibility."""

			pass

		setattr(_ct, "_RemainderColsList", _RemainderColsList)
except Exception:
	# If sklearn isn't available or this fails, let tests handle the resulting errors.
	pass

