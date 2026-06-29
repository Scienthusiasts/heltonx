/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 33 "./CParse/parser.y"

/* doh.h uses #pragma GCC poison with GCC to prevent direct calls to certain
 * standard C library functions being introduced, but those cause errors due
 * to checks like `#if defined YYMALLOC || defined malloc` in the bison
 * template code.  We can't easily arrange to include headers after that
 * template code, so instead we disable the problematic poisoning for this
 * file.
 */
#define DOH_NO_POISON_MALLOC_FREE

#include "swig.h"
#include "cparse.h"
#include "preprocessor.h"
#include <ctype.h>

#define YYMALLOC Malloc
#define YYFREE Free

/* -----------------------------------------------------------------------------
 *                               Externals
 * ----------------------------------------------------------------------------- */

int  yyparse(void);

/* NEW Variables */

static void    *top = 0;      /* Top of the generated parse tree */
static int      unnamed = 0;  /* Unnamed datatype counter */
static Hash    *classes = 0;        /* Hash table of classes */
static Hash    *classes_typedefs = 0; /* Hash table of typedef classes: typedef struct X {...} Y; */
static Symtab  *prev_symtab = 0;
static Node    *current_class = 0;
String  *ModuleName = 0;
static Node    *module_node = 0;
static String  *Classprefix = 0;  
static String  *Namespaceprefix = 0;
static int      inclass = 0;
static Node    *currentOuterClass = 0; /* for nested classes */
static String  *last_cpptype = 0;
static int      inherit_list = 0;
static Parm    *template_parameters = 0;
static int      parsing_template_declaration = 0;
static int      extendmode   = 0;
static int      compact_default_args = 0;
static int      template_reduce = 0;
static int      cparse_externc = 0;
int		ignore_nested_classes = 0;
int		kwargs_supported = 0;

/* -----------------------------------------------------------------------------
 *                            Doxygen Comment Globals
 * ----------------------------------------------------------------------------- */
static String *currentDeclComment = NULL; /* Comment of C/C++ declaration. */

/* -----------------------------------------------------------------------------
 *                            Assist Functions
 * ----------------------------------------------------------------------------- */


 
/* Called by the parser (yyparse) when an error is found.*/
static void yyerror (const char *e) {
  (void)e;
}

static Node *new_node(const_String_or_char_ptr tag) {
  Node *n = Swig_cparse_new_node(tag);
  return n;
}

/* Copies a node.  Does not copy tree links or symbol table data (except for
   sym:name) */

static Node *copy_node(Node *n) {
  Node *nn;
  Iterator k;
  nn = NewHash();
  Setfile(nn,Getfile(n));
  Setline(nn,Getline(n));
  for (k = First(n); k.key; k = Next(k)) {
    String *ci;
    String *key = k.key;
    char *ckey = Char(key);
    if ((strcmp(ckey,"nextSibling") == 0) ||
	(strcmp(ckey,"previousSibling") == 0) ||
	(strcmp(ckey,"parentNode") == 0) ||
	(strcmp(ckey,"lastChild") == 0)) {
      continue;
    }
    if (Strncmp(key,"csym:",5) == 0) continue;
    /* We do copy sym:name.  For templates */
    if ((strcmp(ckey,"sym:name") == 0) || 
	(strcmp(ckey,"sym:weak") == 0) ||
	(strcmp(ckey,"sym:typename") == 0)) {
      String *ci = Copy(k.item);
      Setattr(nn,key, ci);
      Delete(ci);
      continue;
    }
    if (strcmp(ckey,"sym:symtab") == 0) {
      Setattr(nn,"sym:needs_symtab", "1");
    }
    /* We don't copy any other symbol table attributes */
    if (strncmp(ckey,"sym:",4) == 0) {
      continue;
    }
    /* If children.  We copy them recursively using this function */
    if (strcmp(ckey,"firstChild") == 0) {
      /* Copy children */
      Node *cn = k.item;
      while (cn) {
	Node *copy = copy_node(cn);
	appendChild(nn,copy);
	Delete(copy);
	cn = nextSibling(cn);
      }
      continue;
    }
    /* We don't copy the symbol table.  But we drop an attribute 
       requires_symtab so that functions know it needs to be built */

    if (strcmp(ckey,"symtab") == 0) {
      /* Node defined a symbol table. */
      Setattr(nn,"requires_symtab","1");
      continue;
    }
    /* Can't copy nodes */
    if (strcmp(ckey,"node") == 0) {
      continue;
    }
    if ((strcmp(ckey,"parms") == 0) || (strcmp(ckey,"pattern") == 0) || (strcmp(ckey,"throws") == 0)
	|| (strcmp(ckey,"kwargs") == 0)) {
      ParmList *pl = CopyParmList(k.item);
      Setattr(nn,key,pl);
      Delete(pl);
      continue;
    }
    if (strcmp(ckey,"nested:outer") == 0) { /* don't copy outer classes links, they will be updated later */
      Setattr(nn, key, k.item);
      continue;
    }
    /* defaultargs will be patched back in later in update_defaultargs() */
    if (strcmp(ckey,"defaultargs") == 0) {
      Setattr(nn, "needs_defaultargs", "1");
      continue;
    }
    /* same for abstracts, which contains pointers to the source node children, and so will need to be patch too */
    if (strcmp(ckey,"abstracts") == 0) {
      SetFlag(nn, "needs_abstracts");
      continue;
    }
    /* Looks okay.  Just copy the data using Copy */
    ci = Copy(k.item);
    Setattr(nn, key, ci);
    Delete(ci);
  }
  return nn;
}

static void set_comment(Node *n, String *comment) {
  String *name;
  Parm *p;
  if (!n || !comment)
    return;

  if (Getattr(n, "doxygen"))
    Append(Getattr(n, "doxygen"), comment);
  else {
    Setattr(n, "doxygen", comment);
    /* This is the first comment, populate it with @params, if any */
    p = Getattr(n, "parms");
    while (p) {
      if (Getattr(p, "doxygen"))
	Printv(comment, "\n@param ", Getattr(p, "name"), Getattr(p, "doxygen"), NIL);
      p=nextSibling(p);
    }
  }
  
  /* Append same comment to every generated overload */
  name = Getattr(n, "name");
  if (!name)
    return;
  n = nextSibling(n);
  while (n && Getattr(n, "name") && Strcmp(Getattr(n, "name"), name) == 0) {
    Setattr(n, "doxygen", comment);
    n = nextSibling(n);
  }
}

/* -----------------------------------------------------------------------------
 *                              Variables
 * ----------------------------------------------------------------------------- */

static char  *typemap_lang = 0;    /* Current language setting */

static int cplus_mode  = 0;

/* C++ modes */

#define  CPLUS_PUBLIC    1
#define  CPLUS_PRIVATE   2
#define  CPLUS_PROTECTED 3

/* storage classes */

#define SWIG_STORAGE_CLASS_EXTERNC	0x0001
#define SWIG_STORAGE_CLASS_EXTERNCPP	0x0002
#define SWIG_STORAGE_CLASS_EXTERN	0x0004
#define SWIG_STORAGE_CLASS_STATIC	0x0008
#define SWIG_STORAGE_CLASS_TYPEDEF	0x0010
#define SWIG_STORAGE_CLASS_VIRTUAL	0x0020
#define SWIG_STORAGE_CLASS_FRIEND	0x0040
#define SWIG_STORAGE_CLASS_EXPLICIT	0x0080
#define SWIG_STORAGE_CLASS_CONSTEXPR	0x0100
#define SWIG_STORAGE_CLASS_THREAD_LOCAL	0x0200

/* Test if multiple bits are set in x. */
static int multiple_bits_set(unsigned x) { return (x & (x - 1)) != 0; }

static const char* storage_class_string(int c) {
  switch (c) {
    case SWIG_STORAGE_CLASS_EXTERNC:
      return "extern \"C\"";
    case SWIG_STORAGE_CLASS_EXTERNCPP:
      return "extern \"C++\"";
    case SWIG_STORAGE_CLASS_EXTERN:
      return "extern";
    case SWIG_STORAGE_CLASS_STATIC:
      return "static";
    case SWIG_STORAGE_CLASS_TYPEDEF:
      return "typedef";
    case SWIG_STORAGE_CLASS_VIRTUAL:
      return "virtual";
    case SWIG_STORAGE_CLASS_FRIEND:
      return "friend";
    case SWIG_STORAGE_CLASS_EXPLICIT:
      return "explicit";
    case SWIG_STORAGE_CLASS_CONSTEXPR:
      return "constexpr";
    case SWIG_STORAGE_CLASS_THREAD_LOCAL:
      return "thread_local";
  }
  assert(0);
  return "<unknown>";
}

/* include types */
static int   import_mode = 0;

void SWIG_typemap_lang(const char *tm_lang) {
  typemap_lang = Swig_copy_string(tm_lang);
}

void SWIG_cparse_set_compact_default_args(int defargs) {
  compact_default_args = defargs;
}

int SWIG_cparse_template_reduce(int treduce) {
  template_reduce = treduce;
  return treduce;  
}

/* -----------------------------------------------------------------------------
 *                           Assist functions
 * ----------------------------------------------------------------------------- */

static int promote_type(int t) {
  if (t <= T_UCHAR || t == T_CHAR || t == T_WCHAR) return T_INT;
  return t;
}

/* Perform type-promotion for binary operators */
static int promote(int t1, int t2) {
  t1 = promote_type(t1);
  t2 = promote_type(t2);
  return t1 > t2 ? t1 : t2;
}

static String *yyrename = 0;

/* Forward renaming operator */

static String *resolve_create_node_scope(String *cname, int is_class_definition, int *errored);


Hash *Swig_cparse_features(void) {
  static Hash   *features_hash = 0;
  if (!features_hash) features_hash = NewHash();
  return features_hash;
}

/* -----------------------------------------------------------------------------
 * feature_identifier_fix()
 *
 * If a template, return template with all template parameters fully resolved.
 *
 * This is a copy and modification of typemap_identifier_fix.
 * ----------------------------------------------------------------------------- */

static String *feature_identifier_fix(String *s) {
  String *tp = SwigType_istemplate_templateprefix(s);
  if (tp) {
    String *ts, *ta, *tq;
    ts = SwigType_templatesuffix(s);
    ta = SwigType_templateargs(s);
    tq = Swig_symbol_type_qualify(ta,0);
    Append(tp,tq);
    Append(tp,ts);
    Delete(ts);
    Delete(ta);
    Delete(tq);
    return tp;
  } else {
    return NewString(s);
  }
}

static void set_access_mode(Node *n) {
  if (cplus_mode == CPLUS_PUBLIC)
    Setattr(n, "access", "public");
  else if (cplus_mode == CPLUS_PROTECTED)
    Setattr(n, "access", "protected");
  else
    Setattr(n, "access", "private");
}

static void restore_access_mode(Node *n) {
  String *mode = Getattr(n, "access");
  if (Strcmp(mode, "private") == 0)
    cplus_mode = CPLUS_PRIVATE;
  else if (Strcmp(mode, "protected") == 0)
    cplus_mode = CPLUS_PROTECTED;
  else
    cplus_mode = CPLUS_PUBLIC;
}

/* Generate the symbol table name for an object */
/* This is a bit of a mess. Need to clean up */
static String *add_oldname = 0;



static String *make_name(Node *n, String *name,SwigType *decl) {
  String *made_name = 0;
  int destructor = name && (*(Char(name)) == '~');

  if (yyrename) {
    String *s = NewString(yyrename);
    Delete(yyrename);
    yyrename = 0;
    if (destructor  && (*(Char(s)) != '~')) {
      Insert(s,0,"~");
    }
    return s;
  }

  if (!name) return 0;

  if (parsing_template_declaration)
    SetFlag(n, "parsing_template_declaration");
  made_name = Swig_name_make(n, Namespaceprefix, name, decl, add_oldname);
  Delattr(n, "parsing_template_declaration");

  return made_name;
}

/* Generate an unnamed identifier */
static String *make_unnamed(void) {
  unnamed++;
  return NewStringf("$unnamed%d$",unnamed);
}

static int is_operator(String *name) {
  return Strncmp(name,"operator ", 9) == 0;
}

/* Add declaration list to symbol table */
static int  add_only_one = 0;

static void add_symbols(Node *n) {
  String *decl;
  String *wrn = 0;

  if (inclass && n) {
    cparse_normalize_void(n);
  }
  while (n) {
    String *symname = 0;
    String *old_prefix = 0;
    Symtab *old_scope = 0;
    int isfriend = inclass && Checkattr(n, "storage", "friend");
    int iscdecl = Cmp(nodeType(n),"cdecl") == 0;
    int only_csymbol = 0;
    
    if (inclass) {
      String *name = Getattr(n, "name");
      if (isfriend) {
	/* For friends, set the scope to the same as the class that the friend is defined/declared in, that is, pop scope once */
	String *prefix = name ? Swig_scopename_prefix(name) : 0;
	old_prefix = Namespaceprefix;
	old_scope = Swig_symbol_popscope();
	Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	if (!prefix) {
	  /* To check - this should probably apply to operators too */
	  if (name && !is_operator(name) && Namespaceprefix) {
	    String *friendusing = NewStringf("using namespace %s;", Namespaceprefix);
	    Setattr(n, "friendusing", friendusing);
	    Delete(friendusing);
	  }
	} else {
	  /* Qualified friend declarations should not be possible as they are ignored in the parse tree */
	  /* TODO: uncomment out for swig-4.3.0
	  assert(0);
	  */
	}
      } else if (Equal(nodeType(n), "using")) {
	String *uname = Getattr(n, "uname");
	Node *cls = current_class ? current_class : currentOuterClass; /* Current class seems to vary depending on whether it is a template class or a plain class */
	String *nprefix = 0;
	String *nlast = 0;
	Swig_scopename_split(uname, &nprefix, &nlast);
	if (Swig_item_in_list(Getattr(cls, "baselist"), nprefix) || Swig_item_in_list(Getattr(cls, "protectedbaselist"), nprefix) || Swig_item_in_list(Getattr(cls, "privatebaselist"), nprefix)) {
	  String *plain_name = SwigType_istemplate(nprefix) ? SwigType_templateprefix(nprefix) : nprefix;
	  if (Equal(nlast, plain_name)) {
	    /* Using declaration looks like it is using a constructor in an immediate base class - change the constructor name for this class.
	     * C++11 requires using declarations for inheriting base constructors to be in the immediate base class.
	     * Note that we don't try and look up the constructor in the base class as the constructor may be an implicit/implied constructor and hence not exist. */
	    Symtab *stab = Swig_symbol_current();
	    String *nname = Getattr(stab, "name");
	    Setattr(n, "name", nname);
	    SetFlag(n, "usingctor");
	  }
	}
      } else {
	/* for member functions, we need to remove the redundant
	   class scope if provided, as in
	   
	   struct Foo {
	   int Foo::method(int a);
	   };
	   
	*/
	String *prefix = name ? Swig_scopename_prefix(name) : 0;
	if (prefix) {
	  if (Classprefix && (Equal(prefix,Classprefix))) {
	    String *base = Swig_scopename_last(name);
	    Setattr(n,"name",base);
	    Delete(base);
	  }
	  Delete(prefix);
	}
      }
    }

    if (!isfriend && (inclass || extendmode)) {
      Setattr(n,"ismember","1");
    }

    if (extendmode) {
      if (!Getattr(n, "template"))
        SetFlag(n,"isextendmember");
    }

    if (!isfriend && inclass) {
      if ((cplus_mode != CPLUS_PUBLIC)) {
	only_csymbol = 1;
	if (cplus_mode == CPLUS_PROTECTED) {
	  Setattr(n,"access", "protected");
	  only_csymbol = !Swig_need_protected(n);
	} else {
	  Setattr(n,"access", "private");
	  /* private are needed only when they are pure virtuals - why? */
	  if ((Cmp(Getattr(n,"storage"),"virtual") == 0) && (Cmp(Getattr(n,"value"),"0") == 0)) {
	    only_csymbol = 0;
	  }
	  if (Cmp(nodeType(n),"destructor") == 0) {
	    /* Needed for "unref" feature */
	    only_csymbol = 0;
	  }
	}
      } else {
	Setattr(n, "access", "public");
      }
    } else if (extendmode && !inclass) {
      Setattr(n, "access", "public");
    }

    if (Getattr(n,"sym:name")) {
      n = nextSibling(n);
      continue;
    }
    decl = Getattr(n,"decl");
    if (!SwigType_isfunction(decl)) {
      String *name = Getattr(n,"name");
      String *makename = Getattr(n,"parser:makename");
      if (iscdecl) {	
	String *storage = Getattr(n, "storage");
	if (Cmp(storage,"typedef") == 0) {
	  Setattr(n,"kind","typedef");
	} else {
	  SwigType *type = Getattr(n,"type");
	  String *value = Getattr(n,"value");
	  Setattr(n,"kind","variable");
	  if (value && Len(value)) {
	    Setattr(n,"hasvalue","1");
	  }
	  if (type) {
	    SwigType *ty;
	    SwigType *tmp = 0;
	    if (decl) {
	      ty = tmp = Copy(type);
	      SwigType_push(ty,decl);
	    } else {
	      ty = type;
	    }
	    if (storage && (Strstr(storage, "constexpr") || (Strstr(storage, "static") && !SwigType_ismutable(ty)))) {
	      SetFlag(n, "hasconsttype");
	    }
	    Delete(tmp);
	  }
	  if (!type) {
	    Printf(stderr,"notype name %s\n", name);
	  }
	}
      }
      Swig_features_get(Swig_cparse_features(), Namespaceprefix, name, 0, n);
      if (makename) {
	symname = make_name(n, makename,0);
        Delattr(n,"parser:makename"); /* temporary information, don't leave it hanging around */
      } else {
        makename = name;
	symname = make_name(n, makename,0);
      }
      
      if (!symname) {
	symname = Copy(Getattr(n,"unnamed"));
      }
      if (symname) {
	if (parsing_template_declaration)
	  SetFlag(n, "parsing_template_declaration");
	wrn = Swig_name_warning(n, Namespaceprefix, symname,0);
	Delattr(n, "parsing_template_declaration");
      }
    } else {
      String *name = Getattr(n,"name");
      SwigType *fdecl = Copy(decl);
      SwigType *fun = SwigType_pop_function(fdecl);
      if (iscdecl) {	
	Setattr(n,"kind","function");
      }
      
      Swig_features_get(Swig_cparse_features(),Namespaceprefix,name,fun,n);

      symname = make_name(n, name,fun);
      if (parsing_template_declaration)
	SetFlag(n, "parsing_template_declaration");
      wrn = Swig_name_warning(n, Namespaceprefix,symname,fun);
      Delattr(n, "parsing_template_declaration");
      
      Delete(fdecl);
      Delete(fun);
      
    }
    if (!symname) {
      n = nextSibling(n);
      continue;
    }

    if (GetFlag(n, "valueignored")) {
      SWIG_WARN_NODE_BEGIN(n);
      Swig_warning(WARN_PARSE_ASSIGNED_VALUE, Getfile(n), Getline(n), "Value assigned to %s not used due to limited parsing implementation.\n", SwigType_namestr(Getattr(n, "name")));
      SWIG_WARN_NODE_END(n);
    }

    if (cparse_cplusplus) {
      String *value = Getattr(n, "value");
      if (value && Strcmp(value, "delete") == 0) {
	/* C++11 deleted definition / deleted function */
        SetFlag(n,"deleted");
        SetFlag(n,"feature:ignore");
      }
      if (SwigType_isrvalue_reference(Getattr(n, "refqualifier"))) {
	/* Ignore rvalue ref-qualifiers by default
	 * Use Getattr instead of GetFlag to handle explicit ignore and explicit not ignore */
	if (!(Getattr(n, "feature:ignore") || Strncmp(symname, "$ignore", 7) == 0)) {
	  SWIG_WARN_NODE_BEGIN(n);
	  Swig_warning(WARN_TYPE_RVALUE_REF_QUALIFIER_IGNORED, Getfile(n), Getline(n),
	      "Method with rvalue ref-qualifier %s ignored.\n", Swig_name_decl(n));
	  SWIG_WARN_NODE_END(n);
	  SetFlag(n, "feature:ignore");
	}
      }
      if (Equal(Getattr(n, "type"), "auto")) {
	/* Ignore functions with an auto return type and no trailing return type
	 * Use Getattr instead of GetFlag to handle explicit ignore and explicit not ignore */
	if (!(Getattr(n, "feature:ignore") || Strncmp(symname, "$ignore", 7) == 0)) {
	  SWIG_WARN_NODE_BEGIN(n);
	  if (SwigType_isfunction(Getattr(n, "decl")))
	    Swig_warning(WARN_CPP14_AUTO, Getfile(n), Getline(n), "Unable to deduce auto return type for '%s' (ignored).\n", Swig_name_decl(n));
	  else
	    Swig_warning(WARN_CPP11_AUTO, Getfile(n), Getline(n), "Unable to deduce auto type for variable '%s' (ignored).\n", Swig_name_decl(n));
	  SWIG_WARN_NODE_END(n);
	  SetFlag(n, "feature:ignore");
	}
      }
    }
    if (only_csymbol || GetFlag(n, "feature:ignore") || Strncmp(symname, "$ignore", 7) == 0) {
      /* Only add to C symbol table and continue */
      Swig_symbol_add(0, n);
      if (!only_csymbol && !GetFlag(n, "feature:ignore")) {
	/* Print the warning attached to $ignore name, if any */
        char *c = Char(symname) + 7;
	if (strlen(c)) {
	  SWIG_WARN_NODE_BEGIN(n);
	  Swig_warning(0,Getfile(n), Getline(n), "%s\n",c+1);
	  SWIG_WARN_NODE_END(n);
	}
	/* If the symbol was ignored via "rename" and is visible, set also feature:ignore*/
	SetFlag(n, "feature:ignore");
      }
      if (!GetFlag(n, "feature:ignore") && Strcmp(symname,"$ignore") == 0) {
	/* Add feature:ignore if the symbol was explicitly ignored, regardless of visibility */
	SetFlag(n, "feature:ignore");
      }
    } else {
      Node *c;
      if ((wrn) && (Len(wrn))) {
	String *metaname = symname;
	if (!Getmeta(metaname,"already_warned")) {
	  SWIG_WARN_NODE_BEGIN(n);
	  Swig_warning(0,Getfile(n),Getline(n), "%s\n", wrn);
	  SWIG_WARN_NODE_END(n);
	  Setmeta(metaname,"already_warned","1");
	}
      }
      c = Swig_symbol_add(symname,n);

      if (c != n) {
	/* symbol conflict attempting to add in the new symbol */
	if (Getattr(n,"sym:weak")) {
	  Setattr(n,"sym:name",symname);
	} else {
	  Swig_symbol_conflict_warn(n, c, symname, inclass);
        }
      }
    }
    /* restore the class scope if needed */
    if (isfriend) {
      Swig_symbol_setscope(old_scope);
      if (old_prefix) {
	Delete(Namespaceprefix);
	Namespaceprefix = old_prefix;
      }
    }
    Delete(symname);

    if (add_only_one) return;
    n = nextSibling(n);
  }
}


/* add symbols a parse tree node copy */

static void add_symbols_copy(Node *n) {
  String *name;
  int    emode = 0;
  while (n) {
    char *cnodeType = Char(nodeType(n));

    if (strcmp(cnodeType,"access") == 0) {
      String *kind = Getattr(n,"kind");
      if (Strcmp(kind,"public") == 0) {
	cplus_mode = CPLUS_PUBLIC;
      } else if (Strcmp(kind,"private") == 0) {
	cplus_mode = CPLUS_PRIVATE;
      } else if (Strcmp(kind,"protected") == 0) {
	cplus_mode = CPLUS_PROTECTED;
      }
      n = nextSibling(n);
      continue;
    }

    add_oldname = Getattr(n,"sym:name");
    if ((add_oldname) || (Getattr(n,"sym:needs_symtab"))) {
      int old_inclass = -1;
      Node *old_current_class = 0;
      if (add_oldname) {
	DohIncref(add_oldname);
	/*  Disable this, it prevents %rename to work with templates */
	/* If already renamed, we used that name  */
	/*
	if (Strcmp(add_oldname, Getattr(n,"name")) != 0) {
	  Delete(yyrename);
	  yyrename = Copy(add_oldname);
	}
	*/
      }
      Delattr(n,"sym:needs_symtab");
      Delattr(n,"sym:name");

      add_only_one = 1;
      add_symbols(n);

      if (Getattr(n,"partialargs")) {
	Swig_symbol_cadd(Getattr(n,"partialargs"),n);
      }
      add_only_one = 0;
      name = Getattr(n,"name");
      if (Getattr(n,"requires_symtab")) {
	Swig_symbol_newscope();
	Swig_symbol_setscopename(name);
	Delete(Namespaceprefix);
	Namespaceprefix = Swig_symbol_qualifiedscopename(0);
      }
      if (strcmp(cnodeType,"class") == 0) {
	old_inclass = inclass;
	inclass = 1;
	old_current_class = current_class;
	current_class = n;
	if (Strcmp(Getattr(n,"kind"),"class") == 0) {
	  cplus_mode = CPLUS_PRIVATE;
	} else {
	  cplus_mode = CPLUS_PUBLIC;
	}
      }
      if (strcmp(cnodeType,"extend") == 0) {
	emode = cplus_mode;
	cplus_mode = CPLUS_PUBLIC;
      }
      add_symbols_copy(firstChild(n));
      if (strcmp(cnodeType,"extend") == 0) {
	cplus_mode = emode;
      }
      if (Getattr(n,"requires_symtab")) {
	Setattr(n,"symtab", Swig_symbol_popscope());
	Delattr(n,"requires_symtab");
	Delete(Namespaceprefix);
	Namespaceprefix = Swig_symbol_qualifiedscopename(0);
      }
      if (add_oldname) {
	Delete(add_oldname);
	add_oldname = 0;
      }
      if (strcmp(cnodeType,"class") == 0) {
	inclass = old_inclass;
	current_class = old_current_class;
      }
    } else {
      if (strcmp(cnodeType,"extend") == 0) {
	emode = cplus_mode;
	cplus_mode = CPLUS_PUBLIC;
      }
      add_symbols_copy(firstChild(n));
      if (strcmp(cnodeType,"extend") == 0) {
	cplus_mode = emode;
      }
    }
    n = nextSibling(n);
  }
}

/* Add in the "defaultargs" attribute for functions in instantiated templates.
 * n should be any instantiated template (class or start of linked list of functions). */
static void update_defaultargs(Node *n) {
  if (n) {
    Node *firstdefaultargs = n;
    update_defaultargs(firstChild(n));
    n = nextSibling(n);
    /* recursively loop through nodes of all types, but all we really need are the overloaded functions */
    while (n) {
      update_defaultargs(firstChild(n));
      if (!Getattr(n, "defaultargs")) {
	if (Getattr(n, "needs_defaultargs")) {
	  Setattr(n, "defaultargs", firstdefaultargs);
	  Delattr(n, "needs_defaultargs");
	} else {
	  firstdefaultargs = n;
	}
      } else {
	/* Functions added in with %extend (for specialized template classes) will already have default args patched up */
	assert(Getattr(n, "defaultargs") == firstdefaultargs);
      }
      n = nextSibling(n);
    }
  }
}

/* Check a set of declarations to see if any are pure-abstract */

static List *pure_abstracts(Node *n) {
  List *abstracts = 0;
  while (n) {
    if (Cmp(nodeType(n),"cdecl") == 0) {
      String *decl = Getattr(n,"decl");
      if (SwigType_isfunction(decl)) {
	String *init = Getattr(n,"value");
	if (Cmp(init,"0") == 0) {
	  if (!abstracts) {
	    abstracts = NewList();
	  }
	  Append(abstracts,n);
	  SetFlag(n,"abstract");
	}
      }
    } else if (Cmp(nodeType(n),"destructor") == 0) {
      if (Cmp(Getattr(n,"value"),"0") == 0) {
	if (!abstracts) {
	  abstracts = NewList();
	}
	Append(abstracts,n);
	SetFlag(n,"abstract");
      }
    }
    n = nextSibling(n);
  }
  return abstracts;
}

/* Recompute the "abstracts" attribute for the classes in instantiated templates, similarly to update_defaultargs() above. */
static void update_abstracts(Node *n) {
  for (; n; n = nextSibling(n)) {
    Node* const child = firstChild(n);
    if (!child)
      continue;

    update_abstracts(child);

    if (Getattr(n, "needs_abstracts")) {
      Setattr(n, "abstracts", pure_abstracts(child));
      Delattr(n, "needs_abstracts");
    }
  }
}

/* Make a classname */

static String *make_class_name(String *name) {
  String *nname = 0;
  String *prefix;
  if (Namespaceprefix) {
    nname= NewStringf("%s::%s", Namespaceprefix, name);
  } else {
    nname = NewString(name);
  }
  prefix = SwigType_istemplate_templateprefix(nname);
  if (prefix) {
    String *args, *qargs;
    args   = SwigType_templateargs(nname);
    qargs  = Swig_symbol_type_qualify(args,0);
    Append(prefix,qargs);
    Delete(nname);
    Delete(args);
    Delete(qargs);
    nname = prefix;
  }
  return nname;
}

/* Use typedef name as class name */

static void add_typedef_name(Node *n, Node *declnode, String *oldName, Symtab *cscope, String *scpname) {
  String *class_rename = 0;
  SwigType *decl = Getattr(declnode, "decl");
  if (!decl || !Len(decl)) {
    String *cname;
    String *tdscopename;
    String *class_scope = Swig_symbol_qualifiedscopename(cscope);
    String *name = Getattr(declnode, "name");
    cname = Copy(name);
    Setattr(n, "tdname", cname);
    tdscopename = class_scope ? NewStringf("%s::%s", class_scope, name) : Copy(name);
    class_rename = Getattr(n, "class_rename");
    if (class_rename && (Strcmp(class_rename, oldName) == 0))
      Setattr(n, "class_rename", NewString(name));
    if (!classes_typedefs) classes_typedefs = NewHash();
    if (!Equal(scpname, tdscopename) && !Getattr(classes_typedefs, tdscopename)) {
      Setattr(classes_typedefs, tdscopename, n);
    }
    Setattr(n, "decl", decl);
    Delete(class_scope);
    Delete(cname);
    Delete(tdscopename);
  }
}

/* If the class name is qualified.  We need to create or lookup namespace entries */

static Symtab *set_scope_to_global(void) {
  Symtab *symtab = Swig_symbol_global_scope();
  Swig_symbol_setscope(symtab);
  return symtab;
}
 
/* Remove the block braces, { and }, if the 'noblock' attribute is set.
 * Node *kw can be either a Hash or Parmlist. */
static String *remove_block(Node *kw, const String *inputcode) {
  String *modified_code = 0;
  while (kw) {
   String *name = Getattr(kw,"name");
   if (name && (Cmp(name,"noblock") == 0)) {
     char *cstr = Char(inputcode);
     int len = Len(inputcode);
     if (len && cstr[0] == '{') {
       --len; ++cstr; 
       if (len && cstr[len - 1] == '}') { --len; }
       /* we now remove the extra spaces */
       while (len && isspace((int)cstr[0])) { --len; ++cstr; }
       while (len && isspace((int)cstr[len - 1])) { --len; }
       modified_code = NewStringWithSize(cstr, len);
       break;
     }
   }
   kw = nextSibling(kw);
  }
  return modified_code;
}

/*
#define RESOLVE_DEBUG 1
*/
static Node *nscope = 0;
static Node *nscope_inner = 0;

/* Remove the scope prefix from cname and return the base name without the prefix.
 * The scopes required for the symbol name are resolved and/or created, if required.
 * For example AA::BB::CC as input returns CC and creates the namespace AA then inner 
 * namespace BB in the current scope. */
static String *resolve_create_node_scope(String *cname_in, int is_class_definition, int *errored) {
  Symtab *gscope = 0;
  Node *cname_node = 0;
  String *cname = cname_in;
  String *last = Swig_scopename_last(cname);
  nscope = 0;
  nscope_inner = 0;  
  *errored = 0;

  if (Strncmp(cname, "::", 2) == 0) {
    if (is_class_definition) {
      Swig_error(cparse_file, cparse_line, "Using the unary scope operator :: in class definition '%s' is invalid.\n", SwigType_namestr(cname));
      *errored = 1;
      return last;
    }
    cname = NewString(Char(cname) + 2);
  }
  if (is_class_definition) {
    /* Only lookup symbols which are in scope via a using declaration but not via a using directive.
       For example find y via 'using x::y' but not y via a 'using namespace x'. */
    cname_node = Swig_symbol_clookup_no_inherit(cname, 0);
    if (!cname_node) {
      Node *full_lookup_node = Swig_symbol_clookup(cname, 0);
      if (full_lookup_node) {
       /* This finds a symbol brought into scope via both a using directive and a using declaration. */
	Node *last_node = Swig_symbol_clookup_no_inherit(last, 0);
	if (last_node == full_lookup_node)
	  cname_node = last_node;
      }
    }
  } else {
    /* For %template, the template needs to be in scope via any means. */
    cname_node = Swig_symbol_clookup(cname, 0);
  }
#if RESOLVE_DEBUG
  if (!cname_node)
    Printf(stdout, "symbol does not yet exist (%d): [%s]\n", is_class_definition, cname_in);
  else
    Printf(stdout, "symbol does exist (%d): [%s]\n", is_class_definition, cname_in);
#endif

  if (cname_node) {
    /* The symbol has been defined already or is in another scope.
       If it is a weak symbol, it needs replacing and if it was brought into the current scope,
       the scope needs adjusting appropriately for the new symbol.
       Similarly for defined templates. */
    Symtab *symtab = Getattr(cname_node, "sym:symtab");
    Node *sym_weak = Getattr(cname_node, "sym:weak");
    if ((symtab && sym_weak) || Equal(nodeType(cname_node), "template")) {
      /* Check if the scope is the current scope */
      String *current_scopename = Swig_symbol_qualifiedscopename(0);
      String *found_scopename = Swig_symbol_qualifiedscopename(symtab);
      if (!current_scopename)
	current_scopename = NewString("");
      if (!found_scopename)
	found_scopename = NewString("");

      {
	int fail = 1;
	List *current_scopes = Swig_scopename_tolist(current_scopename);
	List *found_scopes = Swig_scopename_tolist(found_scopename);
        Iterator cit = First(current_scopes);
	Iterator fit = First(found_scopes);
#if RESOLVE_DEBUG
Printf(stdout, "comparing current: [%s] found: [%s]\n", current_scopename, found_scopename);
#endif
	for (; fit.item && cit.item; fit = Next(fit), cit = Next(cit)) {
	  String *current = cit.item;
	  String *found = fit.item;
#if RESOLVE_DEBUG
	  Printf(stdout, "  looping %s %s\n", current, found);
#endif
	  if (Strcmp(current, found) != 0)
	    break;
	}

	if (!cit.item) {
	  String *subscope = NewString("");
	  for (; fit.item; fit = Next(fit)) {
	    if (Len(subscope) > 0)
	      Append(subscope, "::");
	    Append(subscope, fit.item);
	  }
	  if (Len(subscope) > 0)
	    cname = NewStringf("%s::%s", subscope, last);
	  else
	    cname = Copy(last);
#if RESOLVE_DEBUG
	  Printf(stdout, "subscope to create: [%s] cname: [%s]\n", subscope, cname);
#endif
	  fail = 0;
	  Delete(subscope);
	} else {
	  if (is_class_definition) {
	    if (!fit.item) {
	      /* It is valid to define a new class with the same name as one forward declared in a parent scope */
	      fail = 0;
	    } else if (Swig_scopename_check(cname)) {
	      /* Classes defined with scope qualifiers must have a matching forward declaration in matching scope */
	      fail = 1;
	    } else {
	      /* This may let through some invalid cases */
	      fail = 0;
	    }
#if RESOLVE_DEBUG
	    Printf(stdout, "scope for class definition, fail: %d\n", fail);
#endif
	  } else {
#if RESOLVE_DEBUG
	    Printf(stdout, "no matching base scope for template\n");
#endif
	    fail = 1;
	  }
	}

	Delete(found_scopes);
	Delete(current_scopes);

	if (fail) {
	  String *cname_resolved = NewStringf("%s::%s", found_scopename, last);
	  Swig_error(cparse_file, cparse_line, "'%s' resolves to '%s' and was incorrectly instantiated in scope '%s' instead of within scope '%s'.\n",
	    SwigType_namestr(cname_in), SwigType_namestr(cname_resolved), SwigType_namestr(current_scopename), SwigType_namestr(found_scopename));
	  *errored = 1;
	  Delete(cname_resolved);
	}
      }

      Delete(current_scopename);
      Delete(found_scopename);
    }
  } else if (!is_class_definition) {
    /* A template instantiation requires a template to be found in scope */
    Swig_error(cparse_file, cparse_line, "Template '%s' undefined.\n", SwigType_namestr(cname_in));
    *errored = 1;
  }

  if (*errored)
    return last;

  if (Swig_scopename_check(cname) && !*errored) {
    Node   *ns;
    String *prefix = Swig_scopename_prefix(cname);
    if (Len(prefix) == 0) {
      String *base = Copy(last);
      /* Use the global scope, but we need to add a 'global' namespace.  */
      if (!gscope) gscope = set_scope_to_global();
      /* note that this namespace is not the "unnamed" one,
	 and we don't use Setattr(nscope,"name", ""),
	 because the unnamed namespace is private */
      nscope = new_node("namespace");
      Setattr(nscope,"symtab", gscope);;
      nscope_inner = nscope;
      Delete(last);
      return base;
    }
    /* Try to locate the scope */
    ns = Swig_symbol_clookup(prefix,0);
    if (!ns) {
      Swig_error(cparse_file, cparse_line, "Undefined scope '%s'\n", SwigType_namestr(prefix));
      *errored = 1;
    } else {
      Symtab *nstab = Getattr(ns,"symtab");
      if (!nstab) {
	Swig_error(cparse_file, cparse_line, "'%s' is not defined as a valid scope.\n", SwigType_namestr(prefix));
	*errored = 1;
	ns = 0;
      } else {
	/* Check if the node scope is the current scope */
	String *tname = Swig_symbol_qualifiedscopename(0);
	String *nname = Swig_symbol_qualifiedscopename(nstab);
	if (tname && (Strcmp(tname,nname) == 0)) {
	  ns = 0;
	  cname = Copy(last);
	}
	Delete(tname);
	Delete(nname);
      }
      if (ns) {
	/* we will try to create a new node using the namespaces we
	   can find in the scope name */
	List *scopes = Swig_scopename_tolist(prefix);
	String *sname;
	Iterator si;

	for (si = First(scopes); si.item; si = Next(si)) {
	  Node *ns1,*ns2;
	  sname = si.item;
	  ns1 = Swig_symbol_clookup(sname,0);
	  assert(ns1);
	  if (Strcmp(nodeType(ns1),"namespace") == 0) {
	    if (Getattr(ns1,"alias")) {
	      ns1 = Getattr(ns1,"namespace");
	    }
	  } else {
	    /* now this last part is a class */
	    si = Next(si);
	    /*  or a nested class tree, which is unrolled here */
	    for (; si.item; si = Next(si)) {
	      if (si.item) {
		Printf(sname,"::%s",si.item);
	      }
	    }
	    /* we get the 'inner' class */
	    nscope_inner = Swig_symbol_clookup(sname,0);
	    /* set the scope to the inner class */
	    Swig_symbol_setscope(Getattr(nscope_inner,"symtab"));
	    /* save the last namespace prefix */
	    Delete(Namespaceprefix);
	    Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	    /* and return the node name, including the inner class prefix */
	    break;
	  }
	  /* here we just populate the namespace tree as usual */
	  ns2 = new_node("namespace");
	  Setattr(ns2,"name",sname);
	  Setattr(ns2,"symtab", Getattr(ns1,"symtab"));
	  add_symbols(ns2);
	  Swig_symbol_setscope(Getattr(ns1,"symtab"));
	  Delete(Namespaceprefix);
	  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	  if (nscope_inner) {
	    if (Getattr(nscope_inner,"symtab") != Getattr(ns2,"symtab")) {
	      appendChild(nscope_inner,ns2);
	      Delete(ns2);
	    }
	  }
	  nscope_inner = ns2;
	  if (!nscope) nscope = ns2;
	}
	cname = Copy(last);
	Delete(scopes);
      }
    }
    Delete(prefix);
  }
  Delete(last);

  return cname;
}
 
/* look for simple typedef name in typedef list */
static String *try_to_find_a_name_for_unnamed_structure(const String *storage, Node *decls) {
  String *name = 0;
  Node *n = decls;
  if (storage && Equal(storage, "typedef")) {
    for (; n; n = nextSibling(n)) {
      if (!Len(Getattr(n, "decl"))) {
	name = Copy(Getattr(n, "name"));
	break;
      }
    }
  }
  return name;
}

/* traverse copied tree segment, and update outer class links*/
static void update_nested_classes(Node *n)
{
  Node *c = firstChild(n);
  while (c) {
    if (Getattr(c, "nested:outer"))
      Setattr(c, "nested:outer", n);
    update_nested_classes(c);
    c = nextSibling(c);
  }
}

/* -----------------------------------------------------------------------------
 * nested_forward_declaration()
 * 
 * Nested struct handling for C++ code if the nested classes are disabled.
 * Create the nested class/struct/union as a forward declaration.
 * ----------------------------------------------------------------------------- */

static Node *nested_forward_declaration(const String *storage, const String *kind, String *sname, String *name, Node *cpp_opt_declarators) {
  Node *nn = 0;

  if (sname) {
    /* Add forward declaration of the nested type */
    Node *n = new_node("classforward");
    Setattr(n, "kind", kind);
    Setattr(n, "name", sname);
    Setattr(n, "storage", storage);
    Setattr(n, "sym:weak", "1");
    add_symbols(n);
    nn = n;
  }

  /* Add any variable instances. Also add in any further typedefs of the nested type.
     Note that anonymous typedefs (eg typedef struct {...} a, b;) are treated as class forward declarations */
  if (cpp_opt_declarators) {
    int storage_typedef = (storage && Equal(storage, "typedef"));
    int variable_of_anonymous_type = !sname && !storage_typedef;
    if (!variable_of_anonymous_type) {
      int anonymous_typedef = !sname && storage_typedef;
      Node *n = cpp_opt_declarators;
      SwigType *type = name;
      while (n) {
	Setattr(n, "type", type);
	Setattr(n, "storage", storage);
	if (anonymous_typedef) {
	  Setattr(n, "nodeType", "classforward");
	  Setattr(n, "sym:weak", "1");
	}
	n = nextSibling(n);
      }
      add_symbols(cpp_opt_declarators);

      if (nn) {
	set_nextSibling(nn, cpp_opt_declarators);
      } else {
	nn = cpp_opt_declarators;
      }
    }
  }

  if (!currentOuterClass || !GetFlag(currentOuterClass, "nested")) {
    if (nn && Equal(nodeType(nn), "classforward")) {
      Node *n = nn;
      if (!GetFlag(n, "feature:ignore")) {
	SWIG_WARN_NODE_BEGIN(n);
	Swig_warning(WARN_PARSE_NAMED_NESTED_CLASS, cparse_file, cparse_line, "Nested %s not currently supported (%s ignored)\n", kind, SwigType_namestr(sname ? sname : name));
	SWIG_WARN_NODE_END(n);
      }
    } else {
      Swig_warning(WARN_PARSE_UNNAMED_NESTED_CLASS, cparse_file, cparse_line, "Nested %s not currently supported (ignored).\n", kind);
    }
  }

  return nn;
}


Node *Swig_cparse(File *f) {
  scanner_file(f);
  top = 0;
  yyparse();
  return (Node *)top;
}

static void single_new_feature(const char *featurename, String *val, Hash *featureattribs, char *declaratorid, SwigType *type, ParmList *declaratorparms, String *qualifier) {
  String *fname;
  String *name;
  String *fixname;
  SwigType *t = Copy(type);

  /* Printf(stdout, "single_new_feature: [%s] [%s] [%s] [%s] [%s] [%s]\n", featurename, val, declaratorid, t, ParmList_str_defaultargs(declaratorparms), qualifier); */

  /* Warn about deprecated features */
  if (strcmp(featurename, "nestedworkaround") == 0)
    Swig_warning(WARN_DEPRECATED_NESTED_WORKAROUND, cparse_file, cparse_line, "The 'nestedworkaround' feature is deprecated.\n");

  fname = NewStringf("feature:%s",featurename);
  if (declaratorid) {
    fixname = feature_identifier_fix(declaratorid);
  } else {
    fixname = NewStringEmpty();
  }
  if (Namespaceprefix) {
    name = NewStringf("%s::%s",Namespaceprefix, fixname);
  } else {
    name = fixname;
  }

  if (declaratorparms) Setmeta(val,"parms",declaratorparms);
  if (!Len(t)) t = 0;
  if (t) {
    if (qualifier) SwigType_push(t,qualifier);
    if (SwigType_isfunction(t)) {
      SwigType *decl = SwigType_pop_function(t);
      if (SwigType_ispointer(t)) {
	String *nname = NewStringf("*%s",name);
	Swig_feature_set(Swig_cparse_features(), nname, decl, fname, val, featureattribs);
	Delete(nname);
      } else {
	Swig_feature_set(Swig_cparse_features(), name, decl, fname, val, featureattribs);
      }
      Delete(decl);
    } else if (SwigType_ispointer(t)) {
      String *nname = NewStringf("*%s",name);
      Swig_feature_set(Swig_cparse_features(),nname,0,fname,val, featureattribs);
      Delete(nname);
    }
  } else {
    /* Global feature, that is, feature not associated with any particular symbol */
    Swig_feature_set(Swig_cparse_features(),name,0,fname,val, featureattribs);
  }
  Delete(fname);
  Delete(name);
}

/* Add a new feature to the Hash. Additional features are added if the feature has a parameter list (declaratorparms)
 * and one or more of the parameters have a default argument. An extra feature is added for each defaulted parameter,
 * simulating the equivalent overloaded method. */
static void new_feature(const char *featurename, String *val, Hash *featureattribs, char *declaratorid, SwigType *type, ParmList *declaratorparms, String *qualifier) {

  ParmList *declparms = declaratorparms;

  /* remove the { and } braces if the noblock attribute is set */
  String *newval = remove_block(featureattribs, val);
  val = newval ? newval : val;

  /* Add the feature */
  single_new_feature(featurename, val, featureattribs, declaratorid, type, declaratorparms, qualifier);

  /* Add extra features if there are default parameters in the parameter list */
  if (type) {
    while (declparms) {
      if (ParmList_has_defaultargs(declparms)) {

        /* Create a parameter list for the new feature by copying all
           but the last (defaulted) parameter */
        ParmList* newparms = CopyParmListMax(declparms, ParmList_len(declparms)-1);

        /* Create new declaration - with the last parameter removed */
        SwigType *newtype = Copy(type);
        Delete(SwigType_pop_function(newtype)); /* remove the old parameter list from newtype */
        SwigType_add_function(newtype,newparms);

        single_new_feature(featurename, Copy(val), featureattribs, declaratorid, newtype, newparms, qualifier);
        declparms = newparms;
      } else {
        declparms = 0;
      }
    }
  }
}

/* check if a function declaration is a plain C object */
static int is_cfunction(Node *n) {
  if (!cparse_cplusplus || cparse_externc)
    return 1;
  if (Swig_storage_isexternc(n)) {
    return 1;
  }
  return 0;
}

/* If the Node is a function with parameters, check to see if any of the parameters
 * have default arguments. If so create a new function for each defaulted argument. 
 * The additional functions form a linked list of nodes with the head being the original Node n. */
static void default_arguments(Node *n) {
  Node *function = n;

  if (function) {
    ParmList *varargs = Getattr(function,"feature:varargs");
    if (varargs) {
      /* Handles the %varargs directive by looking for "feature:varargs" and 
       * substituting ... with an alternative set of arguments.  */
      Parm     *p = Getattr(function,"parms");
      Parm     *pp = 0;
      while (p) {
	SwigType *t = Getattr(p,"type");
	if (Strcmp(t,"v(...)") == 0) {
	  if (pp) {
	    ParmList *cv = Copy(varargs);
	    set_nextSibling(pp,cv);
	    Delete(cv);
	  } else {
	    ParmList *cv =  Copy(varargs);
	    Setattr(function,"parms", cv);
	    Delete(cv);
	  }
	  break;
	}
	pp = p;
	p = nextSibling(p);
      }
    }

    /* Do not add in functions if kwargs is being used or if user wants old default argument wrapping
       (one wrapped method per function irrespective of number of default arguments) */
    if (compact_default_args 
	|| is_cfunction(function) 
	|| GetFlag(function,"feature:compactdefaultargs") 
	|| (GetFlag(function,"feature:kwargs") && kwargs_supported)) {
      ParmList *p = Getattr(function,"parms");
      if (p) 
        Setattr(p,"compactdefargs", "1"); /* mark parameters for special handling */
      function = 0; /* don't add in extra methods */
    }
  }

  while (function) {
    ParmList *parms = Getattr(function,"parms");
    if (ParmList_has_defaultargs(parms)) {

      /* Create a parameter list for the new function by copying all
         but the last (defaulted) parameter */
      ParmList* newparms = CopyParmListMax(parms,ParmList_len(parms)-1);

      /* Create new function and add to symbol table */
      {
	SwigType *ntype = Copy(nodeType(function));
	char *cntype = Char(ntype);
        Node *new_function = new_node(ntype);
        SwigType *decl = Copy(Getattr(function,"decl"));
        int constqualifier = SwigType_isconst(decl);
	String *ccode = Copy(Getattr(function,"code"));
	String *cstorage = Copy(Getattr(function,"storage"));
	String *cvalue = Copy(Getattr(function,"value"));
	SwigType *ctype = Copy(Getattr(function,"type"));
	String *cthrow = Copy(Getattr(function,"throw"));

        Delete(SwigType_pop_function(decl)); /* remove the old parameter list from decl */
        SwigType_add_function(decl,newparms);
        if (constqualifier)
          SwigType_add_qualifier(decl,"const");

        Setattr(new_function,"name", Getattr(function,"name"));
        Setattr(new_function,"code", ccode);
        Setattr(new_function,"decl", decl);
        Setattr(new_function,"parms", newparms);
        Setattr(new_function,"storage", cstorage);
        Setattr(new_function,"value", cvalue);
        Setattr(new_function,"type", ctype);
        Setattr(new_function,"throw", cthrow);

	Delete(ccode);
	Delete(cstorage);
	Delete(cvalue);
	Delete(ctype);
	Delete(cthrow);
	Delete(decl);

        {
          Node *throws = Getattr(function,"throws");
	  ParmList *pl = CopyParmList(throws);
          if (throws) Setattr(new_function,"throws",pl);
	  Delete(pl);
        }

        /* copy specific attributes for global (or in a namespace) template functions - these are not class template methods */
        if (strcmp(cntype,"template") == 0) {
          Node *templatetype = Getattr(function,"templatetype");
          Node *symtypename = Getattr(function,"sym:typename");
          Parm *templateparms = Getattr(function,"templateparms");
          if (templatetype) {
	    Node *tmp = Copy(templatetype);
	    Setattr(new_function,"templatetype",tmp);
	    Delete(tmp);
	  }
          if (symtypename) {
	    Node *tmp = Copy(symtypename);
	    Setattr(new_function,"sym:typename",tmp);
	    Delete(tmp);
	  }
          if (templateparms) {
	    Parm *tmp = CopyParmList(templateparms);
	    Setattr(new_function,"templateparms",tmp);
	    Delete(tmp);
	  }
        } else if (strcmp(cntype,"constructor") == 0) {
          /* only copied for constructors as this is not a user defined feature - it is hard coded in the parser */
          if (GetFlag(function,"feature:new")) SetFlag(new_function,"feature:new");
        }

        add_symbols(new_function);
        /* mark added functions as ones with overloaded parameters and point to the parsed method */
        Setattr(new_function,"defaultargs", n);

        /* Point to the new function, extending the linked list */
        set_nextSibling(function, new_function);
	Delete(new_function);
        function = new_function;
	
	Delete(ntype);
      }
    } else {
      function = 0;
    }
  }
}

/* -----------------------------------------------------------------------------
 * mark_nodes_as_extend()
 *
 * Used by the %extend to mark subtypes with "feature:extend".
 * template instances declared within %extend are skipped
 * ----------------------------------------------------------------------------- */

static void mark_nodes_as_extend(Node *n) {
  for (; n; n = nextSibling(n)) {
    if (Getattr(n, "template") && Strcmp(nodeType(n), "class") == 0)
      continue;
    /* Fix me: extend is not a feature. Replace with isextendmember? */
    Setattr(n, "feature:extend", "1");
    mark_nodes_as_extend(firstChild(n));
  }
}

/* -----------------------------------------------------------------------------
 * add_qualifier_to_declarator()
 *
 * Normally the qualifier is pushed on to the front of the type.
 * Adding a qualifier to a pointer to member function is a special case.
 * For example       : typedef double (Cls::*pmf)(void) const;
 * The qualifier is  : q(const).
 * The declarator is : m(Cls).f(void).
 * We need           : m(Cls).q(const).f(void).
 * ----------------------------------------------------------------------------- */

static String *add_qualifier_to_declarator(SwigType *type, SwigType *qualifier) {
  int is_pointer_to_member_function = 0;
  String *decl = Copy(type);
  String *poppedtype = NewString("");
  assert(qualifier);

  while (decl) {
    if (SwigType_ismemberpointer(decl)) {
      String *memberptr = SwigType_pop(decl);
      if (SwigType_isfunction(decl)) {
	is_pointer_to_member_function = 1;
	SwigType_push(decl, qualifier);
	SwigType_push(decl, memberptr);
	Insert(decl, 0, poppedtype);
	Delete(memberptr);
	break;
      } else {
	Append(poppedtype, memberptr);
      }
      Delete(memberptr);
    } else {
      String *popped = SwigType_pop(decl);
      if (!popped)
	break;
      Append(poppedtype, popped);
      Delete(popped);
    }
  }

  if (!is_pointer_to_member_function) {
    Delete(decl);
    decl = Copy(type);
    SwigType_push(decl, qualifier);
  }

  Delete(poppedtype);
  return decl;
}


#line 1642 "CParse/parser.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "parser.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* END  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_ID = 3,                         /* ID  */
  YYSYMBOL_HBLOCK = 4,                     /* HBLOCK  */
  YYSYMBOL_POUND = 5,                      /* POUND  */
  YYSYMBOL_STRING = 6,                     /* STRING  */
  YYSYMBOL_WSTRING = 7,                    /* WSTRING  */
  YYSYMBOL_INCLUDE = 8,                    /* INCLUDE  */
  YYSYMBOL_IMPORT = 9,                     /* IMPORT  */
  YYSYMBOL_INSERT = 10,                    /* INSERT  */
  YYSYMBOL_CHARCONST = 11,                 /* CHARCONST  */
  YYSYMBOL_WCHARCONST = 12,                /* WCHARCONST  */
  YYSYMBOL_NUM_INT = 13,                   /* NUM_INT  */
  YYSYMBOL_NUM_DOUBLE = 14,                /* NUM_DOUBLE  */
  YYSYMBOL_NUM_FLOAT = 15,                 /* NUM_FLOAT  */
  YYSYMBOL_NUM_LONGDOUBLE = 16,            /* NUM_LONGDOUBLE  */
  YYSYMBOL_NUM_UNSIGNED = 17,              /* NUM_UNSIGNED  */
  YYSYMBOL_NUM_LONG = 18,                  /* NUM_LONG  */
  YYSYMBOL_NUM_ULONG = 19,                 /* NUM_ULONG  */
  YYSYMBOL_NUM_LONGLONG = 20,              /* NUM_LONGLONG  */
  YYSYMBOL_NUM_ULONGLONG = 21,             /* NUM_ULONGLONG  */
  YYSYMBOL_NUM_BOOL = 22,                  /* NUM_BOOL  */
  YYSYMBOL_TYPEDEF = 23,                   /* TYPEDEF  */
  YYSYMBOL_TYPE_INT = 24,                  /* TYPE_INT  */
  YYSYMBOL_TYPE_UNSIGNED = 25,             /* TYPE_UNSIGNED  */
  YYSYMBOL_TYPE_SHORT = 26,                /* TYPE_SHORT  */
  YYSYMBOL_TYPE_LONG = 27,                 /* TYPE_LONG  */
  YYSYMBOL_TYPE_FLOAT = 28,                /* TYPE_FLOAT  */
  YYSYMBOL_TYPE_DOUBLE = 29,               /* TYPE_DOUBLE  */
  YYSYMBOL_TYPE_CHAR = 30,                 /* TYPE_CHAR  */
  YYSYMBOL_TYPE_WCHAR = 31,                /* TYPE_WCHAR  */
  YYSYMBOL_TYPE_VOID = 32,                 /* TYPE_VOID  */
  YYSYMBOL_TYPE_SIGNED = 33,               /* TYPE_SIGNED  */
  YYSYMBOL_TYPE_BOOL = 34,                 /* TYPE_BOOL  */
  YYSYMBOL_TYPE_COMPLEX = 35,              /* TYPE_COMPLEX  */
  YYSYMBOL_TYPE_RAW = 36,                  /* TYPE_RAW  */
  YYSYMBOL_TYPE_NON_ISO_INT8 = 37,         /* TYPE_NON_ISO_INT8  */
  YYSYMBOL_TYPE_NON_ISO_INT16 = 38,        /* TYPE_NON_ISO_INT16  */
  YYSYMBOL_TYPE_NON_ISO_INT32 = 39,        /* TYPE_NON_ISO_INT32  */
  YYSYMBOL_TYPE_NON_ISO_INT64 = 40,        /* TYPE_NON_ISO_INT64  */
  YYSYMBOL_LPAREN = 41,                    /* LPAREN  */
  YYSYMBOL_RPAREN = 42,                    /* RPAREN  */
  YYSYMBOL_COMMA = 43,                     /* COMMA  */
  YYSYMBOL_SEMI = 44,                      /* SEMI  */
  YYSYMBOL_EXTERN = 45,                    /* EXTERN  */
  YYSYMBOL_LBRACE = 46,                    /* LBRACE  */
  YYSYMBOL_RBRACE = 47,                    /* RBRACE  */
  YYSYMBOL_PERIOD = 48,                    /* PERIOD  */
  YYSYMBOL_ELLIPSIS = 49,                  /* ELLIPSIS  */
  YYSYMBOL_CONST_QUAL = 50,                /* CONST_QUAL  */
  YYSYMBOL_VOLATILE = 51,                  /* VOLATILE  */
  YYSYMBOL_REGISTER = 52,                  /* REGISTER  */
  YYSYMBOL_STRUCT = 53,                    /* STRUCT  */
  YYSYMBOL_UNION = 54,                     /* UNION  */
  YYSYMBOL_EQUAL = 55,                     /* EQUAL  */
  YYSYMBOL_SIZEOF = 56,                    /* SIZEOF  */
  YYSYMBOL_MODULE = 57,                    /* MODULE  */
  YYSYMBOL_LBRACKET = 58,                  /* LBRACKET  */
  YYSYMBOL_RBRACKET = 59,                  /* RBRACKET  */
  YYSYMBOL_BEGINFILE = 60,                 /* BEGINFILE  */
  YYSYMBOL_ENDOFFILE = 61,                 /* ENDOFFILE  */
  YYSYMBOL_ILLEGAL = 62,                   /* ILLEGAL  */
  YYSYMBOL_CONSTANT = 63,                  /* CONSTANT  */
  YYSYMBOL_RENAME = 64,                    /* RENAME  */
  YYSYMBOL_NAMEWARN = 65,                  /* NAMEWARN  */
  YYSYMBOL_EXTEND = 66,                    /* EXTEND  */
  YYSYMBOL_PRAGMA = 67,                    /* PRAGMA  */
  YYSYMBOL_FEATURE = 68,                   /* FEATURE  */
  YYSYMBOL_VARARGS = 69,                   /* VARARGS  */
  YYSYMBOL_ENUM = 70,                      /* ENUM  */
  YYSYMBOL_CLASS = 71,                     /* CLASS  */
  YYSYMBOL_TYPENAME = 72,                  /* TYPENAME  */
  YYSYMBOL_PRIVATE = 73,                   /* PRIVATE  */
  YYSYMBOL_PUBLIC = 74,                    /* PUBLIC  */
  YYSYMBOL_PROTECTED = 75,                 /* PROTECTED  */
  YYSYMBOL_COLON = 76,                     /* COLON  */
  YYSYMBOL_STATIC = 77,                    /* STATIC  */
  YYSYMBOL_VIRTUAL = 78,                   /* VIRTUAL  */
  YYSYMBOL_FRIEND = 79,                    /* FRIEND  */
  YYSYMBOL_THROW = 80,                     /* THROW  */
  YYSYMBOL_CATCH = 81,                     /* CATCH  */
  YYSYMBOL_EXPLICIT = 82,                  /* EXPLICIT  */
  YYSYMBOL_STATIC_ASSERT = 83,             /* STATIC_ASSERT  */
  YYSYMBOL_CONSTEXPR = 84,                 /* CONSTEXPR  */
  YYSYMBOL_THREAD_LOCAL = 85,              /* THREAD_LOCAL  */
  YYSYMBOL_DECLTYPE = 86,                  /* DECLTYPE  */
  YYSYMBOL_AUTO = 87,                      /* AUTO  */
  YYSYMBOL_NOEXCEPT = 88,                  /* NOEXCEPT  */
  YYSYMBOL_OVERRIDE = 89,                  /* OVERRIDE  */
  YYSYMBOL_FINAL = 90,                     /* FINAL  */
  YYSYMBOL_USING = 91,                     /* USING  */
  YYSYMBOL_NAMESPACE = 92,                 /* NAMESPACE  */
  YYSYMBOL_NATIVE = 93,                    /* NATIVE  */
  YYSYMBOL_INLINE = 94,                    /* INLINE  */
  YYSYMBOL_TYPEMAP = 95,                   /* TYPEMAP  */
  YYSYMBOL_ECHO = 96,                      /* ECHO  */
  YYSYMBOL_APPLY = 97,                     /* APPLY  */
  YYSYMBOL_CLEAR = 98,                     /* CLEAR  */
  YYSYMBOL_SWIGTEMPLATE = 99,              /* SWIGTEMPLATE  */
  YYSYMBOL_FRAGMENT = 100,                 /* FRAGMENT  */
  YYSYMBOL_WARN = 101,                     /* WARN  */
  YYSYMBOL_LESSTHAN = 102,                 /* LESSTHAN  */
  YYSYMBOL_GREATERTHAN = 103,              /* GREATERTHAN  */
  YYSYMBOL_DELETE_KW = 104,                /* DELETE_KW  */
  YYSYMBOL_DEFAULT = 105,                  /* DEFAULT  */
  YYSYMBOL_LESSTHANOREQUALTO = 106,        /* LESSTHANOREQUALTO  */
  YYSYMBOL_GREATERTHANOREQUALTO = 107,     /* GREATERTHANOREQUALTO  */
  YYSYMBOL_EQUALTO = 108,                  /* EQUALTO  */
  YYSYMBOL_NOTEQUALTO = 109,               /* NOTEQUALTO  */
  YYSYMBOL_LESSEQUALGREATER = 110,         /* LESSEQUALGREATER  */
  YYSYMBOL_ARROW = 111,                    /* ARROW  */
  YYSYMBOL_QUESTIONMARK = 112,             /* QUESTIONMARK  */
  YYSYMBOL_TYPES = 113,                    /* TYPES  */
  YYSYMBOL_PARMS = 114,                    /* PARMS  */
  YYSYMBOL_NONID = 115,                    /* NONID  */
  YYSYMBOL_DSTAR = 116,                    /* DSTAR  */
  YYSYMBOL_DCNOT = 117,                    /* DCNOT  */
  YYSYMBOL_TEMPLATE = 118,                 /* TEMPLATE  */
  YYSYMBOL_OPERATOR = 119,                 /* OPERATOR  */
  YYSYMBOL_CONVERSIONOPERATOR = 120,       /* CONVERSIONOPERATOR  */
  YYSYMBOL_PARSETYPE = 121,                /* PARSETYPE  */
  YYSYMBOL_PARSEPARM = 122,                /* PARSEPARM  */
  YYSYMBOL_PARSEPARMS = 123,               /* PARSEPARMS  */
  YYSYMBOL_DOXYGENSTRING = 124,            /* DOXYGENSTRING  */
  YYSYMBOL_DOXYGENPOSTSTRING = 125,        /* DOXYGENPOSTSTRING  */
  YYSYMBOL_CAST = 126,                     /* CAST  */
  YYSYMBOL_LOR = 127,                      /* LOR  */
  YYSYMBOL_LAND = 128,                     /* LAND  */
  YYSYMBOL_OR = 129,                       /* OR  */
  YYSYMBOL_XOR = 130,                      /* XOR  */
  YYSYMBOL_AND = 131,                      /* AND  */
  YYSYMBOL_LSHIFT = 132,                   /* LSHIFT  */
  YYSYMBOL_RSHIFT = 133,                   /* RSHIFT  */
  YYSYMBOL_PLUS = 134,                     /* PLUS  */
  YYSYMBOL_MINUS = 135,                    /* MINUS  */
  YYSYMBOL_STAR = 136,                     /* STAR  */
  YYSYMBOL_SLASH = 137,                    /* SLASH  */
  YYSYMBOL_MODULO = 138,                   /* MODULO  */
  YYSYMBOL_UMINUS = 139,                   /* UMINUS  */
  YYSYMBOL_NOT = 140,                      /* NOT  */
  YYSYMBOL_LNOT = 141,                     /* LNOT  */
  YYSYMBOL_DCOLON = 142,                   /* DCOLON  */
  YYSYMBOL_YYACCEPT = 143,                 /* $accept  */
  YYSYMBOL_program = 144,                  /* program  */
  YYSYMBOL_interface = 145,                /* interface  */
  YYSYMBOL_declaration = 146,              /* declaration  */
  YYSYMBOL_swig_directive = 147,           /* swig_directive  */
  YYSYMBOL_extend_directive = 148,         /* extend_directive  */
  YYSYMBOL_149_1 = 149,                    /* $@1  */
  YYSYMBOL_apply_directive = 150,          /* apply_directive  */
  YYSYMBOL_clear_directive = 151,          /* clear_directive  */
  YYSYMBOL_constant_directive = 152,       /* constant_directive  */
  YYSYMBOL_echo_directive = 153,           /* echo_directive  */
  YYSYMBOL_stringtype = 154,               /* stringtype  */
  YYSYMBOL_fname = 155,                    /* fname  */
  YYSYMBOL_fragment_directive = 156,       /* fragment_directive  */
  YYSYMBOL_include_directive = 157,        /* include_directive  */
  YYSYMBOL_158_2 = 158,                    /* @2  */
  YYSYMBOL_includetype = 159,              /* includetype  */
  YYSYMBOL_inline_directive = 160,         /* inline_directive  */
  YYSYMBOL_insert_directive = 161,         /* insert_directive  */
  YYSYMBOL_module_directive = 162,         /* module_directive  */
  YYSYMBOL_native_directive = 163,         /* native_directive  */
  YYSYMBOL_pragma_directive = 164,         /* pragma_directive  */
  YYSYMBOL_pragma_arg = 165,               /* pragma_arg  */
  YYSYMBOL_pragma_lang = 166,              /* pragma_lang  */
  YYSYMBOL_rename_directive = 167,         /* rename_directive  */
  YYSYMBOL_rename_namewarn = 168,          /* rename_namewarn  */
  YYSYMBOL_feature_directive = 169,        /* feature_directive  */
  YYSYMBOL_stringbracesemi = 170,          /* stringbracesemi  */
  YYSYMBOL_featattr = 171,                 /* featattr  */
  YYSYMBOL_varargs_directive = 172,        /* varargs_directive  */
  YYSYMBOL_varargs_parms = 173,            /* varargs_parms  */
  YYSYMBOL_typemap_directive = 174,        /* typemap_directive  */
  YYSYMBOL_typemap_type = 175,             /* typemap_type  */
  YYSYMBOL_tm_list = 176,                  /* tm_list  */
  YYSYMBOL_tm_tail = 177,                  /* tm_tail  */
  YYSYMBOL_typemap_parm = 178,             /* typemap_parm  */
  YYSYMBOL_types_directive = 179,          /* types_directive  */
  YYSYMBOL_template_directive = 180,       /* template_directive  */
  YYSYMBOL_warn_directive = 181,           /* warn_directive  */
  YYSYMBOL_c_declaration = 182,            /* c_declaration  */
  YYSYMBOL_183_3 = 183,                    /* $@3  */
  YYSYMBOL_c_decl = 184,                   /* c_decl  */
  YYSYMBOL_c_decl_tail = 185,              /* c_decl_tail  */
  YYSYMBOL_initializer = 186,              /* initializer  */
  YYSYMBOL_cpp_alternate_rettype = 187,    /* cpp_alternate_rettype  */
  YYSYMBOL_cpp_lambda_decl = 188,          /* cpp_lambda_decl  */
  YYSYMBOL_lambda_introducer = 189,        /* lambda_introducer  */
  YYSYMBOL_lambda_template = 190,          /* lambda_template  */
  YYSYMBOL_lambda_body = 191,              /* lambda_body  */
  YYSYMBOL_lambda_tail = 192,              /* lambda_tail  */
  YYSYMBOL_193_4 = 193,                    /* $@4  */
  YYSYMBOL_c_enum_key = 194,               /* c_enum_key  */
  YYSYMBOL_c_enum_inherit = 195,           /* c_enum_inherit  */
  YYSYMBOL_c_enum_forward_decl = 196,      /* c_enum_forward_decl  */
  YYSYMBOL_c_enum_decl = 197,              /* c_enum_decl  */
  YYSYMBOL_c_constructor_decl = 198,       /* c_constructor_decl  */
  YYSYMBOL_cpp_declaration = 199,          /* cpp_declaration  */
  YYSYMBOL_cpp_class_decl = 200,           /* cpp_class_decl  */
  YYSYMBOL_201_5 = 201,                    /* @5  */
  YYSYMBOL_202_6 = 202,                    /* @6  */
  YYSYMBOL_cpp_opt_declarators = 203,      /* cpp_opt_declarators  */
  YYSYMBOL_cpp_forward_class_decl = 204,   /* cpp_forward_class_decl  */
  YYSYMBOL_cpp_template_decl = 205,        /* cpp_template_decl  */
  YYSYMBOL_206_7 = 206,                    /* $@7  */
  YYSYMBOL_cpp_template_possible = 207,    /* cpp_template_possible  */
  YYSYMBOL_template_parms = 208,           /* template_parms  */
  YYSYMBOL_templateparameter = 209,        /* templateparameter  */
  YYSYMBOL_templateparameterstail = 210,   /* templateparameterstail  */
  YYSYMBOL_cpp_using_decl = 211,           /* cpp_using_decl  */
  YYSYMBOL_cpp_namespace_decl = 212,       /* cpp_namespace_decl  */
  YYSYMBOL_213_8 = 213,                    /* @8  */
  YYSYMBOL_214_9 = 214,                    /* @9  */
  YYSYMBOL_cpp_members = 215,              /* cpp_members  */
  YYSYMBOL_216_10 = 216,                   /* $@10  */
  YYSYMBOL_217_11 = 217,                   /* $@11  */
  YYSYMBOL_cpp_member_no_dox = 218,        /* cpp_member_no_dox  */
  YYSYMBOL_cpp_member = 219,               /* cpp_member  */
  YYSYMBOL_cpp_constructor_decl = 220,     /* cpp_constructor_decl  */
  YYSYMBOL_cpp_destructor_decl = 221,      /* cpp_destructor_decl  */
  YYSYMBOL_cpp_conversion_operator = 222,  /* cpp_conversion_operator  */
  YYSYMBOL_cpp_catch_decl = 223,           /* cpp_catch_decl  */
  YYSYMBOL_cpp_static_assert = 224,        /* cpp_static_assert  */
  YYSYMBOL_cpp_protection_decl = 225,      /* cpp_protection_decl  */
  YYSYMBOL_cpp_swig_directive = 226,       /* cpp_swig_directive  */
  YYSYMBOL_cpp_vend = 227,                 /* cpp_vend  */
  YYSYMBOL_anonymous_bitfield = 228,       /* anonymous_bitfield  */
  YYSYMBOL_anon_bitfield_type = 229,       /* anon_bitfield_type  */
  YYSYMBOL_storage_class = 230,            /* storage_class  */
  YYSYMBOL_storage_class_list = 231,       /* storage_class_list  */
  YYSYMBOL_storage_class_raw = 232,        /* storage_class_raw  */
  YYSYMBOL_parms = 233,                    /* parms  */
  YYSYMBOL_rawparms = 234,                 /* rawparms  */
  YYSYMBOL_parm_no_dox = 235,              /* parm_no_dox  */
  YYSYMBOL_parm = 236,                     /* parm  */
  YYSYMBOL_valparms = 237,                 /* valparms  */
  YYSYMBOL_rawvalparms = 238,              /* rawvalparms  */
  YYSYMBOL_valptail = 239,                 /* valptail  */
  YYSYMBOL_valparm = 240,                  /* valparm  */
  YYSYMBOL_callparms = 241,                /* callparms  */
  YYSYMBOL_callptail = 242,                /* callptail  */
  YYSYMBOL_def_args = 243,                 /* def_args  */
  YYSYMBOL_parameter_declarator = 244,     /* parameter_declarator  */
  YYSYMBOL_plain_declarator = 245,         /* plain_declarator  */
  YYSYMBOL_declarator = 246,               /* declarator  */
  YYSYMBOL_notso_direct_declarator = 247,  /* notso_direct_declarator  */
  YYSYMBOL_direct_declarator = 248,        /* direct_declarator  */
  YYSYMBOL_abstract_declarator = 249,      /* abstract_declarator  */
  YYSYMBOL_direct_abstract_declarator = 250, /* direct_abstract_declarator  */
  YYSYMBOL_pointer = 251,                  /* pointer  */
  YYSYMBOL_cv_ref_qualifier = 252,         /* cv_ref_qualifier  */
  YYSYMBOL_ref_qualifier = 253,            /* ref_qualifier  */
  YYSYMBOL_type_qualifier = 254,           /* type_qualifier  */
  YYSYMBOL_type_qualifier_raw = 255,       /* type_qualifier_raw  */
  YYSYMBOL_type = 256,                     /* type  */
  YYSYMBOL_rawtype = 257,                  /* rawtype  */
  YYSYMBOL_type_right = 258,               /* type_right  */
  YYSYMBOL_decltype = 259,                 /* decltype  */
  YYSYMBOL_260_12 = 260,                   /* @12  */
  YYSYMBOL_decltypeexpr = 261,             /* decltypeexpr  */
  YYSYMBOL_primitive_type = 262,           /* primitive_type  */
  YYSYMBOL_primitive_type_list = 263,      /* primitive_type_list  */
  YYSYMBOL_type_specifier = 264,           /* type_specifier  */
  YYSYMBOL_definetype = 265,               /* definetype  */
  YYSYMBOL_default_delete = 266,           /* default_delete  */
  YYSYMBOL_deleted_definition = 267,       /* deleted_definition  */
  YYSYMBOL_explicit_default = 268,         /* explicit_default  */
  YYSYMBOL_ename = 269,                    /* ename  */
  YYSYMBOL_constant_directives = 270,      /* constant_directives  */
  YYSYMBOL_optional_ignored_defines = 271, /* optional_ignored_defines  */
  YYSYMBOL_enumlist = 272,                 /* enumlist  */
  YYSYMBOL_enumlist_item = 273,            /* enumlist_item  */
  YYSYMBOL_edecl_with_dox = 274,           /* edecl_with_dox  */
  YYSYMBOL_edecl = 275,                    /* edecl  */
  YYSYMBOL_etype = 276,                    /* etype  */
  YYSYMBOL_expr = 277,                     /* expr  */
  YYSYMBOL_exprmem = 278,                  /* exprmem  */
  YYSYMBOL_exprsimple = 279,               /* exprsimple  */
  YYSYMBOL_valexpr = 280,                  /* valexpr  */
  YYSYMBOL_exprnum = 281,                  /* exprnum  */
  YYSYMBOL_exprcompound = 282,             /* exprcompound  */
  YYSYMBOL_variadic_opt = 283,             /* variadic_opt  */
  YYSYMBOL_inherit = 284,                  /* inherit  */
  YYSYMBOL_raw_inherit = 285,              /* raw_inherit  */
  YYSYMBOL_286_13 = 286,                   /* $@13  */
  YYSYMBOL_base_list = 287,                /* base_list  */
  YYSYMBOL_base_specifier = 288,           /* base_specifier  */
  YYSYMBOL_289_14 = 289,                   /* @14  */
  YYSYMBOL_290_15 = 290,                   /* @15  */
  YYSYMBOL_access_specifier = 291,         /* access_specifier  */
  YYSYMBOL_templcpptype = 292,             /* templcpptype  */
  YYSYMBOL_cpptype = 293,                  /* cpptype  */
  YYSYMBOL_classkey = 294,                 /* classkey  */
  YYSYMBOL_classkeyopt = 295,              /* classkeyopt  */
  YYSYMBOL_opt_virtual = 296,              /* opt_virtual  */
  YYSYMBOL_virt_specifier_seq = 297,       /* virt_specifier_seq  */
  YYSYMBOL_virt_specifier_seq_opt = 298,   /* virt_specifier_seq_opt  */
  YYSYMBOL_class_virt_specifier_opt = 299, /* class_virt_specifier_opt  */
  YYSYMBOL_exception_specification = 300,  /* exception_specification  */
  YYSYMBOL_qualifiers_exception_specification = 301, /* qualifiers_exception_specification  */
  YYSYMBOL_cpp_const = 302,                /* cpp_const  */
  YYSYMBOL_ctor_end = 303,                 /* ctor_end  */
  YYSYMBOL_ctor_initializer = 304,         /* ctor_initializer  */
  YYSYMBOL_mem_initializer_list = 305,     /* mem_initializer_list  */
  YYSYMBOL_mem_initializer = 306,          /* mem_initializer  */
  YYSYMBOL_less_valparms_greater = 307,    /* less_valparms_greater  */
  YYSYMBOL_identifier = 308,               /* identifier  */
  YYSYMBOL_idstring = 309,                 /* idstring  */
  YYSYMBOL_idstringopt = 310,              /* idstringopt  */
  YYSYMBOL_idcolon = 311,                  /* idcolon  */
  YYSYMBOL_idcolontail = 312,              /* idcolontail  */
  YYSYMBOL_idtemplate = 313,               /* idtemplate  */
  YYSYMBOL_idtemplatetemplate = 314,       /* idtemplatetemplate  */
  YYSYMBOL_idcolonnt = 315,                /* idcolonnt  */
  YYSYMBOL_idcolontailnt = 316,            /* idcolontailnt  */
  YYSYMBOL_string = 317,                   /* string  */
  YYSYMBOL_wstring = 318,                  /* wstring  */
  YYSYMBOL_stringbrace = 319,              /* stringbrace  */
  YYSYMBOL_options = 320,                  /* options  */
  YYSYMBOL_kwargs = 321,                   /* kwargs  */
  YYSYMBOL_stringnum = 322                 /* stringnum  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;


/* Second part of user prologue.  */
#line 1782 "./CParse/parser.y"


/* C++ decltype/auto type deduction. */
static SwigType *deduce_type(const struct Define *dtype) {
  Node *n;
  if (!dtype->val) return NULL;
  n = Swig_symbol_clookup(dtype->val, 0);
  if (n) {
    if (Strcmp(nodeType(n),"enumitem") == 0) {
      /* For an enumitem, the "type" attribute gives us the underlying integer
       * type - we want the "type" attribute from the enum itself, which is
       * "parentNode".
       */
      n = Getattr(n, "parentNode");
    }
    return Getattr(n, "type");
  } else if (dtype->type != T_AUTO && dtype->type != T_UNKNOWN) {
    /* Try to deduce the type from the T_* type code. */
    String *deduced_type = NewSwigType(dtype->type);
    if (Len(deduced_type) > 0) return deduced_type;
    Delete(deduced_type);
  }
  return NULL;
}


#line 2025 "CParse/parser.c"


#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int16 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  60
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   6955

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  143
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  180
/* YYNRULES -- Number of rules.  */
#define YYNRULES  619
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  1164

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   397


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX) YY_CAST (yysymbol_kind_t, YYX)

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,  1818,  1818,  1830,  1834,  1837,  1840,  1843,  1846,  1851,
    1860,  1864,  1871,  1876,  1877,  1878,  1879,  1880,  1889,  1905,
    1915,  1916,  1917,  1918,  1919,  1920,  1921,  1922,  1923,  1924,
    1925,  1926,  1927,  1928,  1929,  1930,  1931,  1932,  1933,  1940,
    1940,  2022,  2032,  2046,  2063,  2080,  2097,  2101,  2112,  2121,
    2134,  2141,  2145,  2156,  2165,  2180,  2193,  2193,  2249,  2250,
    2257,  2276,  2308,  2312,  2322,  2327,  2345,  2388,  2395,  2422,
    2428,  2435,  2436,  2439,  2440,  2447,  2493,  2539,  2550,  2553,
    2580,  2586,  2592,  2598,  2606,  2612,  2618,  2624,  2632,  2633,
    2634,  2637,  2642,  2652,  2688,  2689,  2724,  2741,  2749,  2762,
    2784,  2790,  2794,  2797,  2808,  2813,  2826,  2838,  3073,  3083,
    3090,  3091,  3095,  3095,  3128,  3134,  3144,  3156,  3163,  3243,
    3290,  3354,  3395,  3411,  3415,  3440,  3444,  3455,  3458,  3459,
    3460,  3461,  3462,  3463,  3467,  3478,  3484,  3490,  3498,  3504,
    3508,  3511,  3516,  3519,  3519,  3532,  3535,  3538,  3547,  3550,
    3557,  3579,  3608,  3706,  3761,  3762,  3763,  3764,  3765,  3766,
    3775,  3775,  4021,  4021,  4166,  4167,  4179,  4198,  4198,  4402,
    4408,  4414,  4420,  4426,  4427,  4428,  4429,  4432,  4433,  4436,
    4440,  4443,  4449,  4457,  4465,  4488,  4492,  4497,  4508,  4518,
    4549,  4549,  4607,  4607,  4629,  4656,  4673,  4674,  4679,  4674,
    4687,  4688,  4689,  4701,  4702,  4719,  4720,  4721,  4722,  4723,
    4724,  4725,  4726,  4727,  4728,  4729,  4730,  4731,  4732,  4733,
    4734,  4736,  4737,  4741,  4753,  4787,  4827,  4846,  4867,  4889,
    4912,  4935,  4943,  4950,  4957,  4965,  4973,  4976,  4978,  4979,
    4980,  4981,  4982,  4983,  4984,  4985,  4988,  4999,  5010,  5024,
    5027,  5028,  5029,  5030,  5032,  5038,  5080,  5083,  5084,  5092,
    5093,  5103,  5104,  5105,  5106,  5107,  5108,  5109,  5116,  5128,
    5129,  5133,  5138,  5144,  5151,  5158,  5163,  5172,  5180,  5181,
    5187,  5200,  5204,  5207,  5211,  5215,  5243,  5251,  5255,  5258,
    5262,  5265,  5268,  5272,  5284,  5294,  5307,  5311,  5315,  5322,
    5344,  5361,  5380,  5399,  5406,  5414,  5423,  5432,  5436,  5445,
    5456,  5467,  5479,  5489,  5503,  5512,  5522,  5532,  5542,  5554,
    5566,  5579,  5590,  5601,  5613,  5627,  5634,  5642,  5658,  5666,
    5677,  5688,  5699,  5718,  5726,  5743,  5751,  5758,  5765,  5776,
    5788,  5799,  5811,  5822,  5833,  5853,  5874,  5881,  5887,  5895,
    5903,  5912,  5921,  5922,  5931,  5940,  5948,  5956,  5963,  5971,
    5981,  5992,  6003,  6014,  6021,  6028,  6031,  6048,  6066,  6076,
    6083,  6089,  6094,  6101,  6105,  6110,  6117,  6121,  6127,  6131,
    6137,  6138,  6139,  6145,  6151,  6155,  6156,  6160,  6165,  6171,
    6172,  6173,  6174,  6175,  6177,  6180,  6183,  6186,  6186,  6203,
    6206,  6219,  6245,  6246,  6300,  6304,  6308,  6312,  6316,  6320,
    6324,  6328,  6332,  6336,  6340,  6344,  6348,  6352,  6358,  6373,
    6376,  6377,  6381,  6397,  6414,  6415,  6418,  6419,  6423,  6424,
    6434,  6438,  6443,  6449,  6459,  6470,  6481,  6486,  6491,  6492,
    6498,  6506,  6518,  6536,  6537,  6564,  6568,  6572,  6576,  6580,
    6584,  6588,  6592,  6599,  6600,  6601,  6606,  6612,  6625,  6634,
    6639,  6645,  6660,  6678,  6679,  6682,  6692,  6729,  6737,  6745,
    6753,  6762,  6771,  6791,  6813,  6814,  6815,  6816,  6817,  6818,
    6819,  6820,  6821,  6822,  6825,  6829,  6833,  6837,  6841,  6845,
    6849,  6853,  6857,  6861,  6865,  6869,  6873,  6877,  6885,  6895,
    6899,  6903,  6907,  6921,  6927,  6936,  6945,  6949,  6953,  6979,
    6982,  6987,  6990,  6990,  6991,  6994,  7011,  7020,  7020,  7039,
    7039,  7058,  7059,  7060,  7063,  7068,  7075,  7076,  7080,  7086,
    7090,  7094,  7100,  7101,  7106,  7107,  7110,  7113,  7116,  7119,
    7124,  7125,  7130,  7133,  7138,  7144,  7150,  7156,  7162,  7168,
    7176,  7184,  7189,  7196,  7197,  7207,  7218,  7229,  7239,  7249,
    7257,  7269,  7270,  7273,  7274,  7275,  7276,  7279,  7291,  7297,
    7306,  7307,  7308,  7311,  7312,  7313,  7316,  7317,  7320,  7324,
    7328,  7331,  7334,  7337,  7340,  7345,  7349,  7352,  7359,  7365,
    7368,  7373,  7374,  7380,  7384,  7388,  7391,  7394,  7397,  7402,
    7406,  7409,  7412,  7418,  7421,  7424,  7432,  7435,  7436,  7440,
    7443,  7456,  7460,  7465,  7471,  7475,  7480,  7484,  7491,  7492
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "END", "error", "\"invalid token\"", "ID", "HBLOCK", "POUND", "STRING",
  "WSTRING", "INCLUDE", "IMPORT", "INSERT", "CHARCONST", "WCHARCONST",
  "NUM_INT", "NUM_DOUBLE", "NUM_FLOAT", "NUM_LONGDOUBLE", "NUM_UNSIGNED",
  "NUM_LONG", "NUM_ULONG", "NUM_LONGLONG", "NUM_ULONGLONG", "NUM_BOOL",
  "TYPEDEF", "TYPE_INT", "TYPE_UNSIGNED", "TYPE_SHORT", "TYPE_LONG",
  "TYPE_FLOAT", "TYPE_DOUBLE", "TYPE_CHAR", "TYPE_WCHAR", "TYPE_VOID",
  "TYPE_SIGNED", "TYPE_BOOL", "TYPE_COMPLEX", "TYPE_RAW",
  "TYPE_NON_ISO_INT8", "TYPE_NON_ISO_INT16", "TYPE_NON_ISO_INT32",
  "TYPE_NON_ISO_INT64", "LPAREN", "RPAREN", "COMMA", "SEMI", "EXTERN",
  "LBRACE", "RBRACE", "PERIOD", "ELLIPSIS", "CONST_QUAL", "VOLATILE",
  "REGISTER", "STRUCT", "UNION", "EQUAL", "SIZEOF", "MODULE", "LBRACKET",
  "RBRACKET", "BEGINFILE", "ENDOFFILE", "ILLEGAL", "CONSTANT", "RENAME",
  "NAMEWARN", "EXTEND", "PRAGMA", "FEATURE", "VARARGS", "ENUM", "CLASS",
  "TYPENAME", "PRIVATE", "PUBLIC", "PROTECTED", "COLON", "STATIC",
  "VIRTUAL", "FRIEND", "THROW", "CATCH", "EXPLICIT", "STATIC_ASSERT",
  "CONSTEXPR", "THREAD_LOCAL", "DECLTYPE", "AUTO", "NOEXCEPT", "OVERRIDE",
  "FINAL", "USING", "NAMESPACE", "NATIVE", "INLINE", "TYPEMAP", "ECHO",
  "APPLY", "CLEAR", "SWIGTEMPLATE", "FRAGMENT", "WARN", "LESSTHAN",
  "GREATERTHAN", "DELETE_KW", "DEFAULT", "LESSTHANOREQUALTO",
  "GREATERTHANOREQUALTO", "EQUALTO", "NOTEQUALTO", "LESSEQUALGREATER",
  "ARROW", "QUESTIONMARK", "TYPES", "PARMS", "NONID", "DSTAR", "DCNOT",
  "TEMPLATE", "OPERATOR", "CONVERSIONOPERATOR", "PARSETYPE", "PARSEPARM",
  "PARSEPARMS", "DOXYGENSTRING", "DOXYGENPOSTSTRING", "CAST", "LOR",
  "LAND", "OR", "XOR", "AND", "LSHIFT", "RSHIFT", "PLUS", "MINUS", "STAR",
  "SLASH", "MODULO", "UMINUS", "NOT", "LNOT", "DCOLON", "$accept",
  "program", "interface", "declaration", "swig_directive",
  "extend_directive", "$@1", "apply_directive", "clear_directive",
  "constant_directive", "echo_directive", "stringtype", "fname",
  "fragment_directive", "include_directive", "@2", "includetype",
  "inline_directive", "insert_directive", "module_directive",
  "native_directive", "pragma_directive", "pragma_arg", "pragma_lang",
  "rename_directive", "rename_namewarn", "feature_directive",
  "stringbracesemi", "featattr", "varargs_directive", "varargs_parms",
  "typemap_directive", "typemap_type", "tm_list", "tm_tail",
  "typemap_parm", "types_directive", "template_directive",
  "warn_directive", "c_declaration", "$@3", "c_decl", "c_decl_tail",
  "initializer", "cpp_alternate_rettype", "cpp_lambda_decl",
  "lambda_introducer", "lambda_template", "lambda_body", "lambda_tail",
  "$@4", "c_enum_key", "c_enum_inherit", "c_enum_forward_decl",
  "c_enum_decl", "c_constructor_decl", "cpp_declaration", "cpp_class_decl",
  "@5", "@6", "cpp_opt_declarators", "cpp_forward_class_decl",
  "cpp_template_decl", "$@7", "cpp_template_possible", "template_parms",
  "templateparameter", "templateparameterstail", "cpp_using_decl",
  "cpp_namespace_decl", "@8", "@9", "cpp_members", "$@10", "$@11",
  "cpp_member_no_dox", "cpp_member", "cpp_constructor_decl",
  "cpp_destructor_decl", "cpp_conversion_operator", "cpp_catch_decl",
  "cpp_static_assert", "cpp_protection_decl", "cpp_swig_directive",
  "cpp_vend", "anonymous_bitfield", "anon_bitfield_type", "storage_class",
  "storage_class_list", "storage_class_raw", "parms", "rawparms",
  "parm_no_dox", "parm", "valparms", "rawvalparms", "valptail", "valparm",
  "callparms", "callptail", "def_args", "parameter_declarator",
  "plain_declarator", "declarator", "notso_direct_declarator",
  "direct_declarator", "abstract_declarator", "direct_abstract_declarator",
  "pointer", "cv_ref_qualifier", "ref_qualifier", "type_qualifier",
  "type_qualifier_raw", "type", "rawtype", "type_right", "decltype", "@12",
  "decltypeexpr", "primitive_type", "primitive_type_list",
  "type_specifier", "definetype", "default_delete", "deleted_definition",
  "explicit_default", "ename", "constant_directives",
  "optional_ignored_defines", "enumlist", "enumlist_item",
  "edecl_with_dox", "edecl", "etype", "expr", "exprmem", "exprsimple",
  "valexpr", "exprnum", "exprcompound", "variadic_opt", "inherit",
  "raw_inherit", "$@13", "base_list", "base_specifier", "@14", "@15",
  "access_specifier", "templcpptype", "cpptype", "classkey", "classkeyopt",
  "opt_virtual", "virt_specifier_seq", "virt_specifier_seq_opt",
  "class_virt_specifier_opt", "exception_specification",
  "qualifiers_exception_specification", "cpp_const", "ctor_end",
  "ctor_initializer", "mem_initializer_list", "mem_initializer",
  "less_valparms_greater", "identifier", "idstring", "idstringopt",
  "idcolon", "idcolontail", "idtemplate", "idtemplatetemplate",
  "idcolonnt", "idcolontailnt", "string", "wstring", "stringbrace",
  "options", "kwargs", "stringnum", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-1048)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-590)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     483,  5169,  5270,   310,   111,  4615, -1048, -1048, -1048, -1048,
   -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048,
   -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048,
   -1048,    52,   187,   187,    62, -1048, -1048,   -43,   150,  6028,
     281, -1048,   244,  6308,   771,  1427,   771, -1048, -1048, -1048,
    3478, -1048,   281,   150, -1048,   -23, -1048,   256, -1048,  5676,
   -1048,   162, -1048, -1048, -1048,   269, -1048, -1048,    71,   298,
    5777, -1048, -1048,   298,   322,   332,   341, -1048, -1048, -1048,
     361, -1048,   405, -1048, -1048,   175,   416,   430,    79,   476,
     123,  6098,  6098,   503,   541,   579,   551,  6378, -1048, -1048,
   -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048,   298,
   -1048, -1048, -1048, -1048, -1048, -1048,   505, -1048, -1048, -1048,
   -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048,
   -1048, -1048, -1048, -1048, -1048, -1048, -1048,  6168,   750, -1048,
   -1048, -1048, -1048, -1048, -1048, -1048,   242,  2715, -1048, -1048,
   -1048, -1048,   771, -1048,  4200, -1048,  2443,  2987,  3803,    75,
    1574,  2449,   304,   281, -1048, -1048,    41,   348,    41,   396,
    1571,   496, -1048, -1048, -1048, -1048,   182,   302, -1048, -1048,
     585, -1048,    66, -1048,   199, -1048,  6570,   334,   199,   199,
     206,  2041,   582,   240,   424,   182,   182,   199,  5474,  5676,
   -1048,   281,   281,   297, -1048,   265,   603,   182, -1048, -1048,
     199, -1048,   655,  5676,   623,   257,   641,   662,   199,   579,
     655,  5676, -1048, -1048, -1048,  5575,   281, -1048, -1048,   281,
     570,   579,  1888,   103,   210,   199,   492,  1371,   593,  2041,
     281,  2683,   576,   579, -1048,  2072,   182, -1048, -1048,   -23,
      34, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048,
   -1048, -1048, -1048, -1048,  3803,  2641,  3803,  3803,  3803,  3803,
    3803,  3803, -1048,   634, -1048,   687,   714,  1274,  3229,    39,
   -1048,    57, -1048, -1048,   655,   751, -1048,  4313,  2585,  2585,
     725,   727,  1663,   667, -1048, -1048, -1048,   733, -1048, -1048,
   -1048,  3229, -1048, -1048,  6685,  3229,   755,  4313,   238,   281,
     495,   396, -1048, -1048,   238,   495,   396, -1048, -1048,   663,
   -1048, -1048,  5676,  3123, -1048,  5676,  3259,   238,  2857,  3129,
     495,   396, -1048,   691,  1171, -1048, -1048,   -23,   810,  5371,
   -1048,   776, -1048, -1048,   773,   655,   281,   281, -1048, -1048,
      58,   783, -1048, -1048, -1048,   519,    41,   639,  2851, -1048,
   -1048, -1048, -1048,   281,   791,   785,   352,   788,   796, -1048,
     804,   807,   813, -1048,  6238, -1048,   281, -1048,   811,   820,
   -1048,   823,  6098, -1048, -1048,   747, -1048, -1048,  6098, -1048,
   -1048,   825, -1048,   685,   337,   830,   779,   782,   832, -1048,
     471,   846, -1048, -1048,   112,   323,   323,   323,   561,   790,
     853,   138,   238,   238,   857,  5676,   238,  1706,  2031,   794,
    4065,  1471,    35,   844,   584,  4426,  1471, -1048,   880, -1048,
     493,   655,   886, -1048,  3362,   150, -1048,   926,   929,  4285,
     831,    34,  5957,   899, -1048,  2192, -1048, -1048, -1048, -1048,
   -1048, -1048,  2715, -1048, -1048,  3803,  3803,  3803,  3803,  3803,
    3803,  3803,  3803,  3803,  3803,  3803,  3803,  3803,  3803,  3803,
    3803,  3803,  3803,   933,   939, -1048,   445,   445,  1790,   827,
     441, -1048,   451, -1048, -1048,   445,   445,   626,   829,  1098,
    3803, -1048,  5676,  1669,    50,   495, -1048,  5676,  3395,   495,
   -1048,   904, -1048,  6718,   906, -1048,  6751,   495,   238,   495,
     396, -1048,   238,   495,   396, -1048,  4018,   238,   903,  2402,
     495, -1048, -1048, -1048,  5676, -1048,  5676,   331,   912, -1048,
   -1048,   199,  1514, -1048,   910,  5676,   915,   916, -1048,   217,
     118,  1189,   914,  5676,  2041,   921, -1048, -1048,   257,  4728,
     917, -1048,   750,  6098,   927,   922,  5676,   662,   510,   928,
     199,  5676,   512,  5575,   882,  5575, -1048, -1048,  5676, -1048,
    1098,  1229,   238,    55, -1048,   934,   238,   238,  4083,   930,
     333,   885,   888, -1048, -1048,   789, -1048,   329, -1048, -1048,
   -1048,   247,  2579,  6308,   389,   936,   790,   155,   902, -1048,
   -1048, -1048,   905, -1048, -1048, -1048,   944,   946,  3803,  3803,
    3531,  3667,  3939,    56,  3803,  1511,   952,  6238,   687,  1262,
    1262,  1248,  1248,   879,  3093,  5814,  1933,  4611,  2418,  2192,
     895,   895,   706,   706, -1048, -1048, -1048,   967,   968,   829,
     771, -1048, -1048, -1048,   760,   976,   977,   257,  6784,   981,
     658,   829, -1048,    85,   982, -1048,  6817,   393, -1048,   393,
   -1048,   495,   495,   238,   937,  3265,   495,   396,   495,   238,
     238,   495, -1048, -1048, -1048, -1048,   655,  5676,  4841, -1048,
     979, -1048,   337,   984, -1048,   993, -1048, -1048, -1048, -1048,
     655, -1048, -1048, -1048,   995, -1048,  1471,   655, -1048,   983,
     108,   817,   118, -1048,  1471, -1048,   996, -1048, -1048,  4954,
    6238,   945,  5676, -1048,   997, -1048,   907, -1048,    84,   943,
   -1048,  1005,  1008, -1048,   957,   281,  1044,   832,  1019, -1048,
     257,  1471,   342,   238, -1048,  5676,  3803, -1048, -1048, -1048,
   -1048, -1048, -1048,  4271, -1048,   961,  1021, -1048, -1048,  1003,
    1556,  1916,   486, -1048, -1048,  1025, -1048,   839,  2191,  1023,
    3803,  3803,  3229,  3498,  3803,  3803,  3939,  4498,  3803,  1028,
    1037,  3634,  1038, -1048,  1511, -1048,  3803,  3803,  3803, -1048,
   -1048,  1040,  1042, -1048, -1048, -1048,   772, -1048, -1048, -1048,
   -1048,   495,   238,   238,   495,   495,   495,  1043, -1048,   199,
     199,   393,  2191,  5676,   512,  1514,  1365,   199,  1046, -1048,
    1471,  1050, -1048, -1048,  2041,    30, -1048,  6098, -1048,  1048,
     393,   408,   182,   482, -1048,  2715,   192, -1048,   718,  1031,
      71,  6474, -1048, -1048, -1048, -1048, -1048, -1048, -1048,  5847,
   -1048, -1048,  5067,  1051, -1048,  1054,  3770,   809, -1048,   497,
   -1048,  1003, -1048,   105,  1053,   220,  5676,  2851,  1052,  1030,
   -1048,  1064, -1048,  2041, -1048, -1048, -1048,   902, -1048, -1048,
   -1048,   281, -1048, -1048, -1048,  1063,  1039,  1049,  1055,  6689,
   -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048,
   -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048, -1048,
    1071,   972,  2313, -1048, -1048, -1048, -1048, -1048, -1048, -1048,
    4011, -1048,  1082,   874,  1088, -1048,  3229,  3229,  3229,  3803,
    3803, -1048, -1048,  1091,  5338,  1092,  1093, -1048, -1048, -1048,
     495,   495, -1048, -1048, -1048,    41,  1073,  1099, -1048,   655,
    1100, -1048,  1471,  1730,   512, -1048,  1101, -1048,  1103, -1048,
   -1048, -1048,    84, -1048, -1048,    84,  1041, -1048, -1048,   400,
    6238,  5575,  2041,  6238,  2955, -1048, -1048,   809, -1048, -1048,
      41,  5676, -1048,   428, -1048,   182,  1003, -1048,  1087,  4028,
     117, -1048,  1108,  1110,  1111,   818,   281,   613, -1048,  1471,
   -1048,   187,   902, -1048, -1048, -1048, -1048, -1048,  4050, -1048,
    6623, -1048,  1081,  1083,  1084,   182,  1089,  1090,   237,  2191,
   -1048,  3803, -1048, -1048,  3229,  3229, -1048, -1048, -1048,  1114,
   -1048,  1119, -1048,  1120, -1048,  1471, -1048, -1048, -1048, -1048,
   -1048,  1123, -1048,    41,   257,  1069,    83,   142,  4426, -1048,
     486,  1136, -1048, -1048, -1048, -1048, -1048,  3803, -1048,  1471,
    1003, -1048,  1003,   671, -1048,  1135,  1140,  1137,   499, -1048,
   -1048,    41, -1048,   281,  2191, -1048, -1048,  1471,  1144,  3803,
   -1048,  1143,   874, -1048, -1048, -1048,  1147, -1048, -1048,  1149,
   -1048,  2851,  5676,  1153,  1156,    38,  1157, -1048,  1471,  1154,
   -1048,  3229,    41, -1048, -1048, -1048, -1048, -1048,   281, -1048,
   -1048, -1048,   486,   187,  1167,    41,  5676,  4465,  4050, -1048,
   -1048, -1048,  1158,  5676,  5676,  5676,  1160,  1556,   288, -1048,
     486,  1166, -1048, -1048, -1048,   486,  1174, -1048, -1048,  1471,
    1176,  1177,  1182,  5676, -1048,  6238,   428, -1048, -1048,  2191,
   -1048,  1471, -1048,   627,  1471,  1471,  1471,  1185,  1184, -1048,
   -1048, -1048, -1048, -1048,  2851, -1048, -1048, -1048,  1471,   428,
    1187, -1048, -1048, -1048
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int16 yydefact[] =
{
      12,     0,     0,     0,     0,     0,     4,   570,   404,   412,
     405,   406,   409,   410,   407,   408,   391,   411,   390,   413,
     393,   414,   415,   416,   417,   277,   380,   381,   382,   527,
     528,   145,   510,   510,     0,   571,   572,     0,   582,     0,
       0,   278,     0,     0,   378,   295,   385,   396,   389,   401,
     402,   526,     0,   589,   394,   580,     6,     0,     8,   275,
       1,    17,    62,    58,    59,     0,   262,    16,   259,   611,
       0,    78,    79,   611,    74,     0,     0,   261,   263,   264,
       0,   265,     0,   266,   267,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    10,    11,
       9,    13,    20,    21,    22,    23,    24,    25,    26,   611,
      27,    28,    29,    30,    31,    32,     0,    33,    34,    35,
      36,    37,    38,    14,   109,   114,   111,   110,    18,    15,
     154,   155,   156,   157,   158,   159,   117,     0,   255,   257,
     147,   146,   509,   524,   525,   397,     0,   282,   583,   279,
     392,     3,   384,   379,   275,   388,     0,     0,     0,   582,
     510,   510,   372,     0,   298,   276,   295,   307,   295,   352,
     510,   333,   386,   403,   395,   590,     0,     0,   578,     5,
       0,   268,   269,    19,     0,   604,     0,   260,     0,     0,
       0,     0,   383,   589,   533,     0,     0,     0,   275,   275,
     232,     0,     0,     0,   192,   589,     0,     0,    60,    61,
       0,    48,    49,   275,     0,   303,     0,   102,   577,     0,
     108,   275,   130,   129,   131,   180,     0,   134,   128,     0,
     132,     0,     0,     0,     0,     0,   307,     0,   333,     0,
     425,     0,   514,   259,   258,     0,     0,   584,   591,   581,
     570,   606,   461,   462,   474,   475,   476,   477,   478,   479,
     480,   481,   482,   483,     0,     0,     0,     0,     0,     0,
       0,     0,   285,     0,   280,   284,   444,   383,     0,   454,
     463,   443,   453,   464,   455,   460,   387,   275,   510,   510,
       0,     0,   510,   394,   293,   422,   423,   291,   419,   420,
     421,   418,   443,   363,     0,   294,     0,   275,   509,     0,
     309,   354,   356,   325,   509,   308,   353,   355,   370,   371,
     334,   296,   275,     0,   297,   275,     0,   509,   510,   510,
     304,   347,   346,   325,   357,   588,   587,   586,     0,   275,
     271,   270,   574,   573,     0,   575,     0,     0,   603,   112,
     614,     0,    66,    47,    46,     0,   295,   307,     0,   530,
     531,   529,   532,     0,     0,    70,     0,     0,     0,    94,
       0,     0,     0,   187,     0,    12,     0,   190,     0,     0,
      99,     0,     0,   103,   300,   307,   301,    42,     0,   100,
     576,     0,    52,     0,    51,     0,     0,     0,   186,   184,
     526,     0,   169,   133,     0,     0,     0,     0,   573,     0,
       0,     0,     0,     0,     0,   275,     0,     0,     0,   325,
       0,   554,   333,   149,   589,   275,   554,   512,     0,   511,
     395,   260,     0,   398,     0,     0,   579,     0,     0,     0,
     463,     0,     0,     0,   459,   472,   505,   504,   473,   506,
     507,   569,     0,   281,   508,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   605,   510,   510,   510,   394,
       0,   333,     0,   368,   365,   510,   510,     0,   333,   357,
       0,   364,   275,   510,   394,   318,   326,   275,     0,   317,
     369,     0,   342,     0,     0,   361,     0,   314,   509,   306,
     351,   349,   509,   305,   350,   348,   359,     0,     0,     0,
     310,   358,   585,     7,   275,   272,   275,     0,     0,   171,
      12,     0,     0,   610,     0,   275,     0,     0,    73,     0,
       0,     0,     0,     0,     0,     0,   188,   189,   303,     0,
       0,    12,   256,     0,   104,     0,   275,   102,     0,     0,
       0,     0,     0,   180,   167,     0,   179,   181,   275,    56,
       0,     0,     0,     0,    75,     0,     0,     0,     0,     0,
     545,   536,   537,   377,   376,   550,   375,   373,   546,   551,
     553,     0,     0,     0,     0,     0,   394,   295,   535,   162,
     166,   542,   514,   400,   399,   592,   449,   445,   465,     0,
       0,     0,   372,     0,     0,   295,     0,     0,   284,   501,
     500,   496,   497,   502,     0,   495,   494,   490,   491,   489,
     492,   493,   484,   485,   486,   487,   488,   451,   447,     0,
     357,   337,   336,   335,   359,     0,     0,   358,     0,     0,
       0,   325,   327,   357,     0,   330,     0,   344,   343,   366,
     362,   316,   315,     0,     0,     0,   311,   360,   319,     0,
       0,   313,   274,   273,    64,    65,    63,   275,     0,   615,
     616,   619,   618,   612,    44,     0,    43,    39,    72,    69,
      71,   609,    89,   608,     0,    84,   554,   607,    88,     0,
     618,     0,     0,    95,   554,   231,     0,   193,   194,     0,
       0,     0,   275,    41,     0,   101,     0,   597,   595,     0,
      55,     0,     0,   106,     0,     0,   256,   186,     0,    12,
       0,   554,     0,     0,   344,   275,     0,   548,   539,   538,
     552,   374,   121,     0,   138,   140,     0,   148,   150,   429,
     554,     0,     0,   127,   534,   513,   515,   517,     0,     0,
     288,   288,   466,     0,     0,     0,   372,   371,     0,     0,
       0,     0,     0,   458,   295,   283,     0,   288,   288,   338,
     340,     0,     0,   292,   345,   328,     0,   332,   331,   299,
     367,   320,     0,     0,   312,   324,   323,     0,   113,     0,
       0,   344,     0,   275,     0,     0,     0,     0,     0,    86,
     554,     0,   115,   191,     0,   589,    97,     0,    96,     0,
     344,     0,     0,     0,   593,   282,     0,    50,     0,     0,
     259,     0,   173,   174,   177,   176,   168,   175,   178,     0,
     185,   170,     0,     0,    77,     0,     0,   541,   139,     0,
     122,   426,   428,   436,     0,   430,   275,     0,   551,   562,
     153,     0,   126,     0,   123,   125,   118,   535,   522,   521,
     523,     0,   519,   202,   220,     0,     0,     0,     0,   256,
     243,   244,   237,   245,   218,   200,   241,   236,   238,   239,
     240,   242,   219,   215,   216,   203,   210,   209,   213,   212,
       0,   221,     0,   204,   205,   208,   214,   206,   207,   217,
       0,   160,     0,   443,     0,   498,   469,   468,   467,     0,
       0,   499,   456,     0,   503,     0,     0,   339,   341,   329,
     322,   321,   172,   617,   613,   295,     0,     0,    80,   618,
      91,    85,   554,     0,     0,    93,     0,    67,     0,   105,
     302,   598,   596,   602,   601,   600,     0,    53,    54,   295,
       0,   180,     0,     0,     0,    57,    76,   544,   549,   540,
     295,   275,   141,     0,   427,     0,   429,   438,   440,     0,
     429,   432,   431,     0,     0,     0,     0,     0,   119,   554,
     516,   510,   535,   197,   234,   233,   235,   222,     0,   223,
     256,   195,   391,   390,   393,     0,     0,   389,   394,     0,
     450,     0,   287,   446,   471,   470,   457,   452,   448,     0,
      40,     0,    92,     0,    87,   554,    82,    68,    98,   594,
     599,     0,   183,   295,   303,     0,   333,     0,   275,   547,
       0,     0,   143,   142,   137,   439,   437,     0,   151,   554,
     429,   433,   429,     0,   559,     0,   561,   563,     0,   555,
     556,   295,   518,     0,     0,   164,   163,   554,     0,     0,
     211,     0,   443,    45,    90,    81,     0,   107,   182,     0,
     167,     0,   275,     0,     0,     0,     0,   120,   554,     0,
     441,   442,   295,   435,   434,   557,   558,   560,     0,   565,
     567,   568,     0,   510,     0,   295,   275,     0,     0,   289,
      83,   116,     0,   275,   275,   275,     0,   554,     0,   144,
       0,   564,   124,   520,   198,     0,     0,   249,   161,   554,
       0,     0,     0,   275,   224,     0,     0,   152,   566,     0,
     165,   554,   230,     0,   554,   554,   554,     0,     0,   135,
     199,   225,   246,   248,     0,   228,   227,   226,   554,     0,
       0,   229,   136,   247
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
   -1048, -1048,  -348, -1048, -1048, -1048, -1048,    10,    26,    12,
      29,   696, -1048,    32,    40, -1048, -1048, -1048,    42, -1048,
   -1048,    44, -1048, -1048,    47, -1048,    48,  -529,  -655,    54,
   -1048,    61, -1048,  -334,   679,   -69,    65,    67,    88,    97,
   -1048,   511,  -427,  -875,  -163, -1048, -1048, -1048, -1032, -1047,
   -1048,  -124, -1048, -1048, -1048, -1048, -1048,     3, -1048, -1048,
     131,    14,    24, -1048, -1048,  -527,   675,   516,    99, -1048,
   -1048, -1048,  -741, -1048, -1048,  -812, -1048,   526, -1048,   528,
     102, -1048, -1048, -1048,  -253, -1048, -1048,     5, -1048,  1107,
     -56, -1048,  1216,    23,   431, -1048,   640,   812,    59,   191,
     -38,  -558,  -532,   -60,  1348,    20,  -140,   -12,   387,  -586,
     672,   -42, -1048,   -52,    -1,    25,   -47, -1048, -1048,   -71,
    1215, -1048,  -337,  -144, -1048, -1048, -1048,   418,   295,  -896,
   -1048, -1048,   299, -1048,   980, -1048,  -202,  -138,  -472, -1048,
     -27,   680, -1048, -1048, -1048,   406, -1048, -1048, -1048,  -197,
     -46, -1048, -1048,   293,  -560, -1048, -1048,  -541, -1048,   537,
     167, -1048, -1048,   193,    17,   221,  -154, -1048,   764,  -169,
    -135,  1112, -1048,  -404,  1078, -1048,   581,   292,  -156,  -509
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
       0,     4,     5,   100,   101,   102,   802,   880,   881,   882,
     883,   392,   393,   884,   885,   729,   109,   110,   886,   112,
     113,   887,   689,   196,   888,   116,   889,   695,   542,   890,
     368,   891,   379,   216,   389,   217,   892,   893,   894,   895,
     530,   124,   866,   752,   226,   125,   745,   849,   973,  1044,
    1089,    40,   594,   126,   127,   128,   129,   896,  1009,   758,
    1066,   897,   898,   726,   836,   397,   398,   566,   899,   134,
     551,   375,   900,  1064,  1139,   901,   902,   903,   904,   905,
     906,   136,   907,   908,  1142,   909,  1006,   910,   138,   139,
     290,   181,    41,   182,   273,   274,   453,   275,   912,  1012,
     753,   165,   383,   166,   310,   236,   168,   169,   237,   585,
     586,    43,    44,   276,   192,    46,    47,   245,   433,    48,
      49,    50,   297,   342,   299,   300,   423,   852,   853,   854,
     855,   976,   977,  1090,   278,   279,   280,   302,   282,   283,
     332,   428,   429,   598,   755,   756,   871,   992,   872,    51,
      52,   362,   363,   757,   588,   970,   602,   589,   590,  1143,
     860,   987,  1056,  1057,   175,    53,   350,   391,    54,   178,
      55,   249,   719,   824,   284,   285,   698,   189,   351,   683
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      45,    45,   153,   180,   172,   143,   144,   164,   130,   281,
     137,   248,   298,   240,   291,   103,   706,   105,   191,   131,
     737,   536,   214,   346,    42,    57,   228,   549,   400,   132,
     344,   104,   701,   723,   106,   352,   724,   107,    45,   215,
     215,   335,   248,   366,   740,   108,   808,   111,   555,   114,
     227,   229,   115,   117,   380,   148,   235,   772,    45,   118,
     681,   936,   440,   444,   390,   167,   119,   997,   152,   681,
     120,   789,   121,   790,   947,   386,   410,   185,   306,  1115,
     436,   414,   437,   208,  1051,   241,  1136,   473,     7,  1149,
     592,   242,   652,   122,   176,  1040,   156,   652,   768,   146,
    -286,   531,   123,   145,   133,   140,     7,   135,     7,   339,
     286,    60,  1162,   532,   348,   228,  1159,   158,   348,   177,
     319,     7,   691,   141,   185,   209,   411,   211,   321,   185,
     324,   356,   147,   312,   317,    26,    27,    28,  1081,   227,
     347,     7,   369,   370,   411,   438,   277,   291,   311,   316,
     474,   420,   412,    45,  1093,   384,  1094,   381,   331,   355,
    -286,  1001,   692,  -575,   693,   395,   653,   291,   522,  1116,
     272,   733,   569,   809,    35,    36,   148,   147,     7,   421,
      70,   426,   678,  1082,   769,     7,  1102,   770,   997,   186,
     340,   341,    35,    36,    35,    36,   957,    45,    45,   420,
      37,   822,     7,   709,    38,   185,   353,    35,    36,   858,
     751,   357,    45,     7,   298,   935,   923,  1120,    37,   711,
      45,   688,    38,   185,    45,   309,   823,    35,    36,   975,
    1125,   158,   694,    37,   950,   385,   142,   159,   958,   164,
     616,     7,  1050,   309,   151,     7,   233,   201,   399,   234,
     354,   411,   147,    37,   162,   746,   179,    38,   163,   413,
       7,   312,   317,   980,    35,    36,   501,   202,  1071,   504,
    1083,    35,    36,  1084,   162,   938,   311,   316,   162,   411,
     331,  1070,   183,   525,     7,  1022,    45,   969,    35,    36,
      37,   193,   521,   742,    38,   358,   940,   167,   154,    35,
      36,   511,   515,   295,   296,     7,    45,   205,   480,   482,
     184,    58,   487,  -254,   281,   157,   510,   514,   534,   557,
     376,    45,   548,  1104,    45,    37,     7,    35,    36,    38,
     215,    35,    36,   681,   972,   674,   215,   185,    45,   188,
     348,   373,   147,   348,   981,   982,    35,    36,   348,   646,
     309,    59,   374,    37,    26,    27,    28,    38,   743,   575,
     246,   247,   567,   195,   355,   194,   400,   147,   400,   595,
      35,    36,    37,   197,   736,   679,   159,   675,   309,   587,
     349,   842,   198,   561,   587,   160,   844,   699,   161,   322,
     615,    35,    36,   162,   540,   541,    37,   163,  1150,  1135,
      38,   231,   199,     7,   721,   343,   323,  1039,   386,   343,
     343,     7,    35,    36,    45,  1026,   364,   365,   343,     7,
     246,   336,   581,   582,    45,   480,   482,   487,   378,   645,
     646,   343,   170,   748,  1035,   749,   649,   325,    37,   343,
     162,   654,   159,    26,    27,    28,   200,   647,   298,   312,
     317,   277,   605,   408,   326,   156,   343,   583,   511,   515,
     584,   424,   204,   163,   311,   316,   331,   435,   672,  1042,
     673,   207,  1043,   510,   514,   272,   158,   359,   360,   685,
     696,   331,   415,   641,   704,     7,   287,   862,   384,    35,
      36,    45,   415,   642,   142,   361,    45,    35,    36,   323,
     714,   215,  1079,   157,   667,    35,    36,   782,     7,   323,
     645,   731,   728,     7,  -295,    37,   691,   210,   185,    38,
     984,   583,     7,    45,   584,    45,   156,   951,   730,   863,
     864,    37,   865,   415,    45,    38,   497,   600,   971,  -543,
    1100,   292,    45,   972,   218,  1101,   232,   158,  1029,   318,
     323,  1030,   130,   498,   137,    45,   692,   710,   693,   103,
      45,   105,    45,   131,    45,   774,   703,    45,   385,  -543,
     767,    35,    36,   132,  -295,   104,   858,   164,   106,     7,
     847,   107,   219,   601,   722,   185,   399,   781,   399,   108,
     782,   111,   221,   114,    35,    36,   115,   117,   521,    35,
      36,   954,   170,   118,     1,     2,     3,   298,    35,    36,
     119,   521,   334,  1087,   120,   587,   121,   587,   747,   407,
      37,   797,   913,   913,   159,   716,   694,   338,  -424,   717,
    -424,   155,   667,   233,    37,   167,   234,   122,    38,   913,
     913,   162,   810,   933,   934,   163,   123,   405,   133,   377,
     406,   135,   427,   699,   587,   162,   819,  1059,   814,  1060,
    -424,   348,   587,   147,   170,    35,    36,   415,   643,   382,
     781,  1152,   228,  1153,   478,  1122,    45,  -589,  -589,   845,
     535,   130,  1154,   137,   323,   387,   147,   281,   103,   587,
     105,    37,   131,  1137,   493,    38,   227,   323,  1140,   497,
     785,   403,   132,  -589,   104,   388,   500,   106,   587,   420,
     107,    45,   130,   298,   137,  1095,   498,  1096,   108,   103,
     111,   105,   114,   131,   767,   115,   117,   559,   560,   833,
     452,   839,   118,   132,    45,   104,   164,   451,   106,   119,
     834,   107,   407,   120,   746,   121,   942,   937,   948,   108,
     835,   111,   343,   114,   946,   454,   115,   117,   475,   587,
     228,   851,   343,   118,   400,   215,   122,   483,   587,   484,
     119,    29,    30,    66,   120,   123,   121,   133,   587,   718,
     135,   343,   959,   489,   227,   229,   240,   964,   556,    32,
      33,   490,   152,   242,   167,   243,   492,   122,   572,   162,
     983,   287,    45,   989,   150,   323,   123,   516,   133,   171,
     523,   135,   407,   497,   929,   527,   174,  1160,   157,   526,
     914,    26,    27,    28,   277,   533,   613,    77,    78,    79,
     498,   543,    81,   538,    83,    84,   925,   926,   544,  1007,
     539,  1055,   470,   471,   472,   130,   545,   137,   272,   203,
     206,   546,   103,   552,   105,    45,   131,   547,   964,   806,
     807,   230,   553,   851,   242,   554,   132,   558,   104,   579,
    1068,   106,   562,  1072,   107,   565,   170,   580,   581,   582,
     238,   563,   108,  1025,   111,   564,   114,   568,  1151,   115,
     117,  1155,  1156,  1157,   386,   571,   118,  1019,   581,   582,
     587,   574,   421,   119,   426,  1161,   570,   120,  1034,   121,
     578,  1037,   868,   869,   870,  1041,  -290,  1011,   293,  1049,
     593,  1032,   295,   296,   313,   313,   599,   320,   603,   606,
     122,   815,   607,   614,   333,   170,   637,   298,  1067,   123,
     617,   133,   638,   640,   135,   644,   657,   587,   659,   691,
     230,   185,   669,   677,   684,   238,   702,   170,   591,   686,
      45,   708,   687,   597,  1062,   371,   372,   705,   712,   713,
      45,   735,   720,   725,   384,   738,   734,   739,   750,   171,
     754,   427,  1086,   587,   399,   760,   792,   761,   851,   816,
     401,   693,   851,   402,   773,  1078,   409,   313,   313,   318,
     817,   419,   170,   422,   150,   238,   430,   587,   777,   778,
     298,   466,   467,   468,   469,   470,   471,   472,   779,   780,
     343,   343,   799,   784,   787,   587,  1112,   800,   343,   468,
     469,   470,   471,   472,   170,   801,   803,    45,   805,   820,
     812,   171,   952,   953,   955,   825,   587,   826,  1067,   821,
    1126,   479,   481,   481,   385,   827,   488,  1130,  1131,  1132,
     828,   841,   851,   848,   851,   850,    70,    66,   867,   911,
     919,   494,   313,   496,   978,   587,  1123,  1147,   313,   920,
     922,    45,   927,  1148,   928,   932,   960,   587,   943,   830,
     949,   313,   313,   313,   945,   966,   967,   999,   313,   587,
     979,     7,   587,   587,   587,    45,   986,   985,   988,   993,
     528,   529,    45,    45,    45,   994,   587,   170,   998,   409,
    1020,    77,    78,    79,  1010,   995,    81,   537,    83,    84,
    1013,   996,    45,  1016,  1017,  1018,   301,   304,   305,   154,
     550,  1021,  1047,   807,  1031,  1027,   187,  1028,    26,    27,
      28,  1052,  1053,   318,   500,  1054,   157,  -252,  1073,  -251,
    -253,   170,   831,  1074,  1075,  1069,  -250,  1077,   212,   481,
     481,   481,  1080,   220,     7,   573,   313,   313,  1088,  1097,
     313,   313,   313,  1098,   313,  1106,  1099,    35,    36,   596,
    1108,  1110,     7,  1111,  1113,   185,   978,  1114,  1119,  1117,
    1129,  1133,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   411,    37,  1124,  1138,  1141,   159,  1144,  1145,
     517,    26,    27,    28,  1146,   434,   160,  1158,   680,   161,
     972,  1163,     7,   804,   162,   185,   715,   832,   163,  1128,
     727,   811,   639,   840,   439,   244,   445,   446,   447,   448,
     449,   450,   837,   171,   838,   149,   956,   651,   775,   741,
      35,    36,   345,  1109,   618,   173,   345,   345,   843,   974,
     355,  1046,   313,   990,  1045,   345,   313,     7,    35,    36,
     313,   313,   759,   313,  1134,  1063,    37,   859,   345,   337,
      38,  1121,   818,   295,   296,     0,   345,   394,     0,   518,
       0,     0,   519,   503,   238,     0,   506,     0,   238,   404,
     345,   309,   171,   345,     0,   154,     0,  -295,    35,    36,
       0,   431,     0,   155,     0,     0,     0,     0,     0,   156,
       0,     0,   157,     0,   171,   238,   313,     0,   301,     0,
     313,   313,   313,     0,    37,     0,     0,   944,   159,     0,
     158,     0,     0,     0,   455,   456,     0,   233,   459,     0,
     234,     0,     0,    35,    36,   162,     0,     0,     7,   163,
       0,     0,   459,     0,     7,     0,     0,  -295,     0,   171,
     466,   467,   468,   469,   470,   471,   472,     0,     0,    37,
       0,     0,     0,   159,   466,   467,   468,   469,   470,   471,
     472,     0,   160,     0,     0,   161,   355,     0,     0,   941,
     162,   171,   411,     0,   163,     0,     0,   313,     0,     0,
     416,   170,     0,     0,  1085,   407,     0,   313,     0,   313,
       7,     0,     0,   313,   313,   619,   620,   621,   622,   623,
     624,   625,   626,   627,   628,   629,   630,   631,   632,   633,
     634,   635,   636,     0,    35,    36,     0,     0,     0,     0,
      35,    36,     0,     0,     0,     0,   238,     0,   154,     0,
     648,     0,     0,     0,     0,     0,   155,     0,   656,  1023,
      37,     0,   156,     0,   159,   157,    37,     0,     0,   829,
      38,     0,     0,   233,   171,     0,   234,   313,     0,   417,
       0,   162,   418,   158,     0,   163,     0,   230,     0,   315,
       0,   309,     0,     0,     7,     0,    35,    36,   330,     0,
     185,    26,    27,    28,     0,     0,  1061,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,     0,   171,     0,
       0,     0,    37,     0,     0,     0,   159,     0,     0,     0,
       0,   579,   154,     0,     0,   160,   313,   313,   161,   580,
     581,   582,  1076,   162,     0,     0,   156,   163,     0,   157,
     238,     0,   301,     0,     7,     0,     0,     7,   238,     0,
       0,     0,   315,     0,     0,   330,  1092,   158,   762,   763,
     626,   629,   634,     0,   771,   230,     0,   856,     0,   583,
      35,    36,   584,     0,  1105,   676,    26,    27,    28,   345,
     682,   857,   307,     0,     0,   307,     0,   690,   697,   700,
     327,     0,     0,   308,     0,  1118,    37,   238,     0,   157,
     159,     0,   157,     0,     0,   991,   579,     0,   345,   160,
     697,     0,   161,     0,   580,   581,   582,   162,     0,   732,
       0,   163,     0,     0,   859,     0,   495,     0,     0,     0,
      35,    36,   499,    35,    36,     0,     7,     0,     0,     0,
       0,     0,     7,     0,  1008,   507,   509,   513,     0,     0,
       0,     0,   520,     0,   583,     0,    37,   584,     0,    37,
      38,     0,     0,    38,     0,     0,     0,     0,     0,   328,
       0,     0,   329,     0,   154,     0,     0,   238,     0,     7,
     307,   309,   142,     0,   309,     0,   846,     0,   142,     0,
       0,   157,     0,  1033,     0,     0,  1036,   157,   238,     0,
       0,   301,     0,     7,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   238,   916,   917,   448,   411,   918,     0,
    1058,     0,    35,    36,     0,   576,   924,     0,    35,    36,
     495,   499,   238,     0,   507,   509,   513,     0,   520,     0,
       0,   355,     0,     0,  1024,     0,     0,     0,    37,     0,
     697,     0,   159,     0,    37,     0,     0,     0,    38,   697,
       0,   485,     0,     7,   486,    35,    36,   485,   171,     0,
     486,     0,   596,   163,     0,     0,     0,     0,     0,   309,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    35,
      36,    37,     0,     0,     0,    38,     0,  1103,     0,     0,
       0,   287,     0,     0,     0,     0,     0,   301,     0,   142,
       0,   650,     0,     0,     0,    37,   309,     0,   157,   159,
       0,     0,     0,     0,     0,     0,   661,     0,   233,     0,
     662,   234,  1058,     0,   666,   668,   162,   671,     0,     0,
     163,     0,   238,     0,     0,     0,     0,   345,   345,    35,
      36,     0,   697,   939,     0,   345,     0,     0,     0,     0,
       0,     7,     0,     0,   185,     0,     0,     0,     0,  1014,
    1015,     0,     0,     0,     0,    37,     0,     0,   431,    38,
       0,     0,     0,     0,     0,     0,     0,   861,   485,   250,
     650,   486,   185,   251,   661,   662,   666,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,     0,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,   264,     0,     0,
       0,     0,   294,     0,     0,     0,    26,    27,    28,    29,
      30,     0,   265,     0,     0,     0,     0,    35,    36,     0,
       0,     0,     0,     0,     0,     0,    31,    32,    33,     0,
       0,     0,   295,   296,     0,     0,     0,     0,     0,     0,
       0,   786,    34,    37,     0,    35,    36,    38,     0,     0,
       0,   791,     0,   794,     0,     0,   405,   795,   796,   406,
     295,   296,   697,     0,   162,     0,     0,  1091,     0,     0,
       0,    37,     0,     0,     7,    38,     0,     0,     0,   455,
     456,   457,   458,   459,     7,     0,     0,   266,     0,  1107,
     267,   268,   269,     0,     0,     0,   270,   271,     0,     0,
       0,   301,   463,   464,   465,   466,   467,   468,   469,   470,
     471,   472,   411,   432,     0,   250,     0,     0,   185,   251,
     577,   786,   355,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,     0,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,   264,     0,     0,     0,     0,     0,     0,
      35,    36,    26,    27,    28,    29,    30,     0,   265,     0,
      35,    36,     0,     0,   301,     0,     0,     0,     0,     0,
     930,   931,    31,    32,    33,     0,    37,     0,     0,     0,
      38,     0,     0,     0,     0,     0,    37,     0,    34,     0,
     159,    35,    36,     0,     0,     0,     0,     0,     0,   233,
       0,   309,   234,     0,     0,     0,     0,   162,     0,     0,
       0,   163,     0,     0,     0,     0,     0,    37,     0,     0,
       0,    38,   873,     0,  -256,    62,     0,     0,     0,    63,
      64,    65,     0,   266,     0,     0,   267,   268,   269,     0,
       0,     0,   270,   271,    66,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,     0,     0,     0,   874,    68,     0,  -201,     0,
       0,  -256,  -256,  -256,  -256,  -256,     0,     0,     0,     0,
       0,     0,     0,     0,    70,    71,    72,   875,    74,    75,
      76,  -256,  -256,  -256,   876,   877,   878,     0,    77,    78,
      79,     0,    80,    81,    82,    83,    84,  -256,  -256,     0,
    -256,  -256,    85,     0,     0,     0,    89,    90,    91,    92,
      93,    94,    95,     0,     0,     0,     0,     0,   455,   456,
     457,   458,   459,     0,    96,     0,  -256,     0,     0,    97,
    -256,  -256,     0,     0,   873,   879,  -256,    62,     0,     0,
       0,    63,    64,    65,   466,   467,   468,   469,   470,   471,
     472,  -256,     0,     0,     0,     0,    66,  -256,  -256,  -256,
    -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,  -256,  -256,     0,     0,     0,   874,    68,     0,
    -201,     0,     0,  -256,  -256,  -256,  -256,  -256,     0,     0,
       0,     0,     0,     0,     0,     0,    70,    71,    72,   875,
      74,    75,    76,  -256,  -256,  -256,   876,   877,   878,     0,
      77,    78,    79,     0,    80,    81,    82,    83,    84,  -256,
    -256,     0,  -256,  -256,    85,     7,     0,     0,    89,    90,
      91,    92,    93,    94,    95,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    96,     0,  -256,     0,
       0,    97,  -256,  -256,     0,     0,     0,  1000,     0,     0,
       0,     0,     0,   411,     0,     0,   250,     0,     0,   185,
     251,   670,     7,  -256,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,     0,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,   264,     0,     0,     0,     0,   294,
     307,    35,    36,    26,    27,    28,    29,    30,   314,   265,
       0,     0,     0,     0,     0,     0,     0,   157,     0,     0,
       0,     0,     0,    31,    32,    33,     0,    37,     0,     0,
       0,    38,     0,     0,   455,   456,   457,   458,   459,    34,
       0,     0,    35,    36,     0,     0,     0,     0,    35,    36,
       0,     0,   309,     0,     0,     0,     0,   295,   296,   465,
     466,   467,   468,   469,   470,   471,   472,     0,    37,     0,
       0,     0,    38,     0,    37,     0,     0,     0,    38,     0,
       0,     0,     0,     0,   266,     0,     0,   267,   268,   269,
       0,     0,   250,   270,   271,   185,   251,     0,     7,   309,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,     0,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
     264,     0,     0,     0,     0,     0,   154,     0,     0,    26,
      27,    28,    29,    30,   142,   265,     0,   744,     0,     0,
       0,     0,     0,   157,   441,     0,     0,   185,   251,    31,
      32,    33,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,     0,    34,     0,     0,    35,    36,
       0,     0,     0,     0,    35,    36,     0,     0,     0,     0,
       0,     0,   442,   295,   296,     0,     7,     0,     0,     0,
     443,     0,     0,     0,    37,     0,     0,   265,    38,     0,
      37,     0,     0,     0,   159,     0,     0,     0,     0,     0,
     266,     0,     0,   267,   268,   269,     0,     0,   250,   270,
     271,   185,   251,     0,   425,   163,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,     0,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,   264,     0,     0,     0,
       0,     0,     0,     0,    25,    26,    27,    28,    29,    30,
       0,   265,    35,    36,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    31,    32,    33,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    37,     0,
       0,    34,   159,     0,    35,    36,     0,     0,     0,     0,
       0,   233,     0,     0,   234,     0,     0,     0,     0,   162,
       0,     0,     0,   163,     0,     0,     0,     0,     0,     0,
      37,     0,     0,     0,    38,     0,     0,     0,     0,    39,
       0,     0,     0,     0,     0,     0,   266,     0,     0,   267,
     268,   269,     0,     0,   250,   270,   271,   185,   251,     0,
       7,     0,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,     0,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,   264,     0,     0,     0,     0,     0,   307,     0,
       0,    26,    27,    28,    29,    30,   508,   265,     0,     0,
       0,     0,     0,     0,     0,   157,     0,     0,     0,     0,
       0,    31,    32,    33,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    34,     0,     0,
      35,    36,     0,     0,     0,     0,    35,    36,     0,     0,
       0,     0,     0,     0,     0,   295,   296,     0,     7,     0,
       0,     0,     0,     0,     0,     0,    37,     0,     0,     0,
      38,     0,    37,     0,     0,     0,    38,     0,     0,     0,
       0,     0,   266,     0,     0,   267,   268,   269,     0,     0,
     250,   270,   271,   185,   251,     0,  1038,   309,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
       0,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,   264,     0,
       0,     0,     0,     0,     0,     0,     0,    26,    27,    28,
      29,    30,     0,   265,    35,    36,   303,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    31,    32,    33,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      37,     0,     0,    34,   159,     0,    35,    36,     0,     0,
       0,     0,     0,   233,     0,     0,   234,     0,     0,     0,
       0,   162,     0,     0,     0,   163,     0,     0,     0,     0,
       0,     0,    37,     0,     0,     0,    38,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   266,     0,
       0,   267,   268,   269,     0,     0,   250,   270,   271,   185,
     251,     0,     7,     0,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,     0,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,   264,     0,     0,     0,     0,   776,
     307,     0,     0,    26,    27,    28,    29,    30,   512,   265,
       0,     0,   502,     0,     0,     0,     0,   157,     0,     0,
       0,     0,     0,    31,    32,    33,     0,     0,     0,   455,
     456,   457,   458,   459,     0,   460,     0,     0,     0,    34,
       0,     0,    35,    36,     0,     0,     0,     0,    35,    36,
     461,   462,   463,   464,   465,   466,   467,   468,   469,   470,
     471,   472,     0,     0,     0,     0,     0,     0,    37,     0,
       0,     0,    38,     0,    37,     0,     0,     0,    38,     0,
       0,     0,     0,     0,   266,     0,     0,   267,   268,   269,
       0,     0,   250,   270,   271,   185,   251,     0,     7,   309,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,     0,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
     264,     0,     0,     0,     0,     0,   411,     0,     0,    26,
      27,    28,    29,    30,   793,   265,     0,     0,   505,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    31,
      32,    33,     0,     0,     0,   455,   456,   457,   458,   459,
       0,   460,     0,     0,     0,    34,     0,     0,    35,    36,
       0,     0,     0,     0,    35,    36,   461,   462,   463,   464,
     465,   466,   467,   468,   469,   470,   471,   472,     0,     0,
       0,     0,     0,     0,    37,     0,     0,     0,    38,     0,
      37,     0,     0,     0,    38,     0,     0,     0,     0,     0,
     266,     0,     0,   267,   268,   269,     0,     0,   250,   270,
     271,   185,   251,     0,   604,   309,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,     0,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,   264,     0,     0,     0,
       0,     0,     0,     0,     0,    26,    27,    28,    29,    30,
       0,   265,     0,     0,   655,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    31,    32,    33,   455,   456,
     457,   458,   459,     0,   460,     0,     0,     0,     0,     0,
       0,    34,     0,     0,    35,    36,     0,     0,     0,   461,
     462,   463,   464,   465,   466,   467,   468,   469,   470,   471,
     472,     0,     8,     9,    10,    11,    12,    13,    14,    15,
      37,    17,     0,    19,    38,    21,    22,    23,    24,     0,
       0,     0,     0,     0,     0,     0,   266,     0,     0,   267,
     268,   269,     0,     0,   250,   270,   271,   185,   251,     0,
     915,     0,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,     0,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,   264,   764,     0,     0,     0,     0,     0,     0,
       0,    26,    27,    28,    29,    30,     0,   265,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    31,    32,    33,   455,   456,   457,   458,   459,     0,
     460,     0,     0,     0,     0,     0,     0,    34,     0,     0,
      35,    36,     0,     0,     0,   461,   462,   463,   464,   465,
     466,   467,   468,   469,   470,   471,   472,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    37,     0,     0,     0,
      38,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   266,     0,     0,   267,   268,   269,     0,     0,
     250,   270,   271,   185,   251,     0,   921,     0,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
       0,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,   264,   765,
       0,     0,     0,     0,     0,     0,     0,    26,    27,    28,
      29,    30,     0,   265,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    31,    32,    33,
     455,   456,   457,   458,   459,     0,   460,     0,     0,     0,
       0,     0,     0,    34,     0,     0,    35,    36,     0,     0,
       0,   461,   462,   463,   464,   465,   466,   467,   468,   469,
     470,   471,   472,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    37,     0,     0,     0,    38,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   266,     0,
       0,   267,   268,   269,     0,     0,   250,   270,   271,   185,
     251,     0,   968,     0,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,     0,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,   264,     0,     0,     0,     0,     0,
       0,     0,     0,    26,    27,    28,    29,    30,     0,   265,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    31,    32,    33,   455,   456,   457,   458,
     459,     0,   460,     0,     0,     0,     0,     0,     0,    34,
       0,     0,    35,    36,     0,     0,     0,   461,   462,   463,
     464,   465,   466,   467,   468,   469,   470,   471,   472,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    37,     0,
       0,     0,    38,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   266,     0,     0,   267,   268,   269,
       0,     0,   250,   270,   271,   185,   251,     0,     0,     0,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,     0,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
     264,     0,     0,     0,     0,     0,     0,     0,     0,    26,
      27,    28,    29,    30,     0,   265,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    31,
      32,    33,     0,     0,     7,     0,     0,     0,     0,     0,
       0,     7,     0,     0,     0,    34,     0,     0,    35,    36,
       0,     7,     0,     0,     0,     8,     9,    10,    11,    12,
      13,    14,    15,  1002,    17,  1003,    19,  1004,    21,    22,
      23,    24,     0,     7,    37,     0,     0,     0,    38,   307,
       0,    26,    27,    28,    29,    30,     0,   663,     7,   355,
       0,     0,  1048,   267,   268,   766,   157,     0,     0,   270,
     271,    31,    32,    33,     0,     0,     7,     0,     0,     0,
       0,   355,     0,     0,  1065,     0,     0,    34,   239,     0,
      35,    36,     0,     0,     0,     0,   411,    35,    36,     0,
       0,     0,     0,     0,   517,     0,     0,    35,    36,     0,
       0,     0,     0,     0,   411,     0,    37,     0,     0,     0,
      38,   963,   663,    37,     0,     0,     0,    38,     0,    35,
      36,     0,     0,    37,     0,     0,   664,   159,     0,   665,
       0,  1005,     0,     0,    35,    36,   233,     0,   309,   234,
       0,     0,     0,     0,   162,    37,     0,     0,   163,   159,
       0,     0,    35,    36,     0,     0,     0,     0,   233,     0,
      37,   234,     0,     0,    38,     0,   162,     0,     0,     0,
     163,     0,     0,   518,     0,     0,   519,     0,    37,     0,
       0,     0,    38,     7,     0,   309,     0,     0,     0,     0,
       0,   664,     0,     0,   665,     0,     0,     0,     0,     0,
       0,     0,     0,   309,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,   287,     0,     0,     0,     0,     0,     0,     0,    25,
      26,    27,    28,    29,    30,     0,     0,     0,   157,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      31,    32,    33,     0,     7,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    34,     0,     0,    35,
      36,     0,     0,     0,     0,     8,     9,    10,    11,    12,
      13,    14,    15,   222,    17,   223,    19,   224,    21,    22,
      23,    24,     0,     0,     0,    37,     7,     0,     0,    38,
       0,     0,     0,     0,    39,     0,     0,   608,   288,     0,
       0,   289,     0,     0,     0,     0,   162,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,   287,     0,     0,    34,     0,     0,
      35,    36,    25,    26,    27,    28,    29,    30,     0,     0,
       0,   157,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    31,    32,    33,    37,     0,   609,     0,
      38,   455,   456,   457,   458,   459,     0,   460,     0,    34,
       0,     0,    35,    36,     0,     0,     0,     0,     0,     0,
       0,     0,   461,   610,   463,   464,   611,   466,   467,   468,
     469,   612,   471,   472,     0,     0,     0,     0,    37,     7,
       0,     0,    38,     0,     0,     0,     0,    39,     0,     0,
       0,   476,     0,     0,   477,     0,     0,     0,     0,   162,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,     0,     0,     0,
       0,     0,     0,     0,     0,    25,    26,    27,    28,    29,
      30,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    31,    32,    33,     0,
       0,     7,     0,     0,     0,     0,     0,     0,     0,  1127,
       0,     0,    34,     0,     0,    35,    36,     0,     0,     0,
       0,     0,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,     0,
       0,    37,     0,     0,     0,    38,     0,     0,     0,     0,
      39,    29,    30,     0,   405,     0,     0,   406,     0,     0,
       0,     0,   162,     0,     0,     0,     0,     0,    31,    32,
      33,   455,   456,   457,   458,   459,     0,   460,     0,     0,
       0,     0,     0,     0,    34,     0,     0,    35,    36,     0,
       0,     0,   461,   462,   463,   464,   465,   466,   467,   468,
     469,   470,   471,   472,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    37,     0,    -2,    61,    38,  -256,    62,
       0,     0,     0,    63,    64,    65,     0,     0,     0,     0,
       0,     0,     0,     0,   162,     0,     0,     0,    66,  -256,
    -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,  -256,  -256,  -256,  -256,     0,     0,     0,    67,
      68,     0,     0,     0,     0,  -256,  -256,  -256,  -256,  -256,
       0,     0,    69,     0,     0,     0,     0,     0,    70,    71,
      72,    73,    74,    75,    76,  -256,  -256,  -256,     0,     0,
       0,     0,    77,    78,    79,     0,    80,    81,    82,    83,
      84,  -256,  -256,     0,  -256,  -256,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,   455,   456,   457,
     458,   459,     0,     0,     0,     0,     0,     0,    96,    61,
    -256,  -256,    62,    97,  -256,     0,    63,    64,    65,    98,
      99,   464,   465,   466,   467,   468,   469,   470,   471,   472,
       0,    66,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,     0,
       0,     0,    67,    68,     0,   707,     0,     0,  -256,  -256,
    -256,  -256,  -256,     0,     0,    69,     0,     0,     0,     0,
       0,    70,    71,    72,    73,    74,    75,    76,  -256,  -256,
    -256,     0,     0,     0,     0,    77,    78,    79,     0,    80,
      81,    82,    83,    84,  -256,  -256,     0,  -256,  -256,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    96,    61,  -256,  -256,    62,    97,  -256,     0,    63,
      64,    65,    98,    99,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    66,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,     0,     0,     0,    67,    68,     0,   798,     0,
       0,  -256,  -256,  -256,  -256,  -256,     0,     0,    69,     0,
       0,     0,     0,     0,    70,    71,    72,    73,    74,    75,
      76,  -256,  -256,  -256,     0,     0,     0,     0,    77,    78,
      79,     0,    80,    81,    82,    83,    84,  -256,  -256,     0,
    -256,  -256,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    96,    61,  -256,  -256,    62,    97,
    -256,     0,    63,    64,    65,    98,    99,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    66,  -256,  -256,
    -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,  -256,  -256,  -256,     0,     0,     0,    67,    68,
       0,   813,     0,     0,  -256,  -256,  -256,  -256,  -256,     0,
       0,    69,     0,     0,     0,     0,     0,    70,    71,    72,
      73,    74,    75,    76,  -256,  -256,  -256,     0,     0,     0,
       0,    77,    78,    79,     0,    80,    81,    82,    83,    84,
    -256,  -256,     0,  -256,  -256,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    94,    95,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    96,    61,  -256,
    -256,    62,    97,  -256,     0,    63,    64,    65,    98,    99,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      66,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,     0,     0,
       0,    67,    68,     0,     0,     0,     0,  -256,  -256,  -256,
    -256,  -256,     0,     0,    69,     0,     0,     0,   965,     0,
      70,    71,    72,    73,    74,    75,    76,  -256,  -256,  -256,
       0,     0,     0,     0,    77,    78,    79,     0,    80,    81,
      82,    83,    84,  -256,  -256,     0,  -256,  -256,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,     0,
       6,     0,     7,     0,     0,     0,     0,     0,     0,     0,
      96,     0,  -256,     0,     0,    97,  -256,     0,     0,     0,
       0,    98,    99,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
       0,     0,     0,     0,     0,     0,     0,     0,    25,    26,
      27,    28,    29,    30,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    31,
      32,    33,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    34,     0,     0,    35,    36,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    56,     0,     7,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    37,     0,     0,     0,    38,     0,
       0,     0,     0,    39,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,     0,     0,     0,     0,     0,     0,     0,     0,    25,
      26,    27,    28,    29,    30,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      31,    32,    33,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    34,     0,     0,    35,
      36,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     7,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    37,     0,     0,     0,    38,
       0,     0,     0,     0,    39,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,     0,     0,     0,     0,     0,     0,     0,     0,
      25,    26,    27,    28,    29,    30,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    31,    32,    33,   455,   456,   457,   458,   459,     0,
       0,     0,     0,     0,     0,     0,     0,    34,     0,     0,
      35,    36,     0,     0,     0,   461,   462,   463,   464,   465,
     466,   467,   468,   469,   470,   471,   472,     7,     0,     0,
       0,     0,     0,     0,     0,     0,    37,   367,     0,     0,
      38,     0,     0,     0,     0,    39,   524,     0,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,     0,     0,     0,     0,     0,
       0,     0,     0,    25,    26,    27,    28,    29,    30,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    31,    32,    33,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      34,     0,     0,    35,    36,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     7,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    37,
       0,     0,     0,    38,     0,     0,     0,     0,    39,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,     0,     0,     0,     0,
       0,     0,     0,     0,    25,    26,    27,    28,    29,    30,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    31,    32,    33,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    34,     0,     0,    35,    36,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     7,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      37,     0,     0,   396,    38,     0,     0,     0,     0,    39,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,     0,     0,     0,
       0,     0,     0,     0,     0,    25,    26,    27,    28,    29,
      30,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    31,    32,    33,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    34,     0,     0,    35,    36,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   190,     0,
       7,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    37,     0,     0,     0,    38,     0,     0,     0,     0,
      39,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    26,    27,    28,
      29,    30,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    31,    32,    33,
       7,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    34,     0,     0,    35,    36,     0,     0,
       0,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,     0,     0,
       0,     0,    37,     0,     0,     0,    38,    26,    27,    28,
      29,    30,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    31,    32,    33,
     455,   456,   457,   458,   459,     0,     0,     0,     0,     0,
       0,     0,     0,    34,   962,     0,    35,    36,     0,     0,
       0,     0,   462,   463,   464,   465,   466,   467,   468,   469,
     470,   471,   472,     0,     0,     0,     0,     0,     0,     0,
     250,     0,    37,   185,   251,     0,    38,   963,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
       0,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    26,    27,    28,
      29,    30,     0,   265,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    31,    32,    33,
       0,     7,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    34,     0,     0,    35,    36,     0,     0,
       0,     0,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,     0,
       0,     0,    37,     0,     0,     0,    38,    25,    26,    27,
      28,    29,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    31,    32,
      33,     7,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    34,     0,     0,    35,    36,     0,
       0,     0,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,   213,
       0,     0,     0,    37,     0,     0,     0,    38,    26,    27,
      28,    29,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    31,    32,
      33,     7,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    34,     0,     0,    35,    36,     0,
       0,     0,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,     0,
       0,     0,     0,    37,     0,     0,     0,    38,    26,    27,
      28,    29,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    31,    32,
      33,     7,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    34,   239,     0,    35,    36,     0,
       0,     0,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,     0,
       0,     0,     0,    37,     0,     0,     0,    38,    26,    27,
      28,    29,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    31,    32,
      33,     7,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    34,     0,     0,    35,    36,     0,
       0,     0,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,     0,
       0,     0,     0,    37,     0,     0,     0,    38,     0,     0,
       0,    29,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    31,    32,
      33,     7,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    34,     0,     0,    35,    36,     0,
       0,     0,     8,     9,    10,    11,    12,    13,    14,    15,
     222,    17,   223,    19,   224,    21,    22,    23,    24,     0,
       0,     0,     0,    37,     0,     0,     0,    38,     0,     0,
       0,    29,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    32,
      33,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    34,     0,     0,    35,    36,     0,
       0,     0,     0,     0,     0,     0,     0,     7,     0,     0,
     225,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    37,     0,     0,     0,    38,     8,     9,
      10,    11,    12,    13,    14,    15,   222,    17,   223,    19,
     224,    21,    22,    23,    24,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    29,    30,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    32,    33,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      34,     0,     0,    35,    36,     0,     0,     0,     0,     0,
       0,     0,     0,     7,     0,     0,   961,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    37,
       0,     0,     0,    38,     8,     9,    10,    11,    12,    13,
      14,    15,   222,    17,   223,    19,   224,    21,    22,    23,
      24,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    29,    30,     0,     0,    62,     0,     0,
       0,     0,     0,    65,     0,     0,     0,     0,     0,     0,
       0,    32,    33,     0,     0,     0,    66,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    34,     0,     0,    35,
      36,     0,     0,     0,     0,     0,     0,   874,    68,     0,
    -196,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    37,    70,    71,    72,    38,
      74,    75,    76,    62,     0,     0,   876,   877,   878,    65,
      77,    78,    79,     0,    80,    81,    82,    83,    84,     0,
       0,     0,    66,     0,    85,     0,     0,     0,    89,    90,
      91,    92,    93,    94,    95,     0,     0,     0,     0,     0,
       0,     0,     0,   874,    68,     0,    96,     0,     0,     0,
       0,    97,     0,     0,   491,     0,     0,     0,     0,     0,
       0,     0,    70,    71,    72,     0,    74,    75,    76,     0,
       0,     0,   876,   877,   878,     0,    77,    78,    79,     0,
      80,    81,    82,    83,    84,     0,     0,   658,     0,     0,
      85,     0,     0,     0,    89,    90,    91,    92,    93,    94,
      95,   455,   456,   457,   458,   459,     0,   460,     0,     0,
       0,     0,    96,     0,     0,     0,     0,    97,     0,     0,
     660,     0,   461,   462,   463,   464,   465,   466,   467,   468,
     469,   470,   471,   472,   455,   456,   457,   458,   459,     0,
     460,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   783,     0,   461,   462,   463,   464,   465,
     466,   467,   468,   469,   470,   471,   472,   455,   456,   457,
     458,   459,     0,   460,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   788,     0,   461,   462,
     463,   464,   465,   466,   467,   468,   469,   470,   471,   472,
     455,   456,   457,   458,   459,     0,   460,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   461,   462,   463,   464,   465,   466,   467,   468,   469,
     470,   471,   472,   455,   456,   457,   458,   459,     0,   460,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   461,   462,   463,   464,   465,   466,
     467,   468,   469,   470,   471,   472
};

static const yytype_int16 yycheck[] =
{
       1,     2,    44,    59,    46,    32,    33,    45,     5,   147,
       5,   146,   156,   137,   154,     5,   548,     5,    70,     5,
     580,   358,    91,   186,     1,     2,    97,   375,   225,     5,
     184,     5,   541,   562,     5,   189,   563,     5,    39,    91,
      92,   176,   177,   197,   585,     5,   701,     5,   382,     5,
      97,    97,     5,     5,   210,    38,   116,   615,    59,     5,
     532,   802,   264,   265,   218,    45,     5,   879,    43,   541,
       5,   657,     5,   659,    44,   215,   232,     6,     3,    41,
     249,   235,    48,     4,   980,   137,  1118,    48,     3,  1136,
      55,   137,    42,     5,   117,   970,    55,    42,    42,   142,
      43,    43,     5,    41,     5,    53,     3,     5,     3,    43,
     152,     0,  1159,    55,     6,   186,  1148,    76,     6,   142,
     162,     3,     4,    71,     6,    46,    41,     4,   166,     6,
     168,   191,   102,   160,   161,    50,    51,    52,    55,   186,
     186,     3,   198,   199,    41,   111,   147,   287,   160,   161,
     111,   116,    49,   154,  1050,   215,  1052,   213,   170,    41,
     103,   902,    44,    55,    46,   221,   116,   307,   337,   131,
     147,   116,    60,   702,    89,    90,   159,   102,     3,   239,
      63,   241,   530,    41,   128,     3,  1061,   131,  1000,   118,
     124,   125,    89,    90,    89,    90,     4,   198,   199,   116,
     115,   117,     3,   551,   119,     6,     0,    89,    90,   750,
      55,   191,   213,     3,   358,   801,   774,  1092,   115,   553,
     221,     4,   119,     6,   225,   140,   142,    89,    90,   124,
    1105,    76,   114,   115,   820,   215,    49,   119,    46,   277,
     442,     3,   125,   140,     0,     3,   128,    72,   225,   131,
      44,    41,   102,   115,   136,   592,     0,   119,   140,    49,
       3,   288,   289,    43,    89,    90,   322,    92,  1009,   325,
     128,    89,    90,   131,   136,   804,   288,   289,   136,    41,
     292,    44,   120,   339,     3,   940,   287,   847,    89,    90,
     115,    70,   334,    46,   119,    55,   805,   277,    41,    89,
      90,   328,   329,   104,   105,     3,   307,    86,   288,   289,
      41,     1,   292,    76,   452,    58,   328,   329,   356,   388,
      55,   322,   374,  1064,   325,   115,     3,    89,    90,   119,
     382,    89,    90,   805,    46,     4,   388,     6,   339,    41,
       6,    44,   102,     6,   124,   125,    89,    90,     6,   489,
     140,    41,    55,   115,    50,    51,    52,   119,   111,   415,
     118,   119,   400,    41,    41,    73,   563,   102,   565,   425,
      89,    90,   115,    41,    41,   531,   119,    46,   140,   421,
      46,   729,    41,    46,   426,   128,    44,   541,   131,    41,
     442,    89,    90,   136,    42,    43,   115,   140,  1139,   111,
     119,   109,    41,     3,   560,   184,    58,   967,   548,   188,
     189,     3,    89,    90,   415,   944,   195,   196,   197,     3,
     118,   119,    89,    90,   425,   405,   406,   407,   207,   489,
     570,   210,    45,    44,   961,    46,   492,    41,   115,   218,
     136,   497,   119,    50,    51,    52,    41,   489,   592,   476,
     477,   452,   435,   232,    58,    55,   235,   128,   485,   486,
     131,   240,    46,   140,   476,   477,   478,   246,   524,    41,
     526,    41,    44,   485,   486,   452,    76,    53,    54,   535,
     540,   493,    41,    42,   544,     3,    41,     1,   548,    89,
      90,   492,    41,    42,    49,    71,   497,    89,    90,    58,
     556,   553,  1034,    58,   516,    89,    90,   647,     3,    58,
     570,   571,   568,     3,    43,   115,     4,    41,     6,   119,
     857,   128,     3,   524,   131,   526,    55,   119,   570,    43,
      44,   115,    46,    41,   535,   119,    41,    44,    41,    46,
      41,   154,   543,    46,    41,    46,    41,    76,   952,   162,
      58,   955,   549,    58,   549,   556,    44,   552,    46,   549,
     561,   549,   563,   549,   565,   617,   543,   568,   548,    76,
     612,    89,    90,   549,   103,   549,  1117,   615,   549,     3,
     743,   549,    41,    90,   561,     6,   563,   647,   565,   549,
     730,   549,    41,   549,    89,    90,   549,   549,   640,    89,
      90,   119,   215,   549,   121,   122,   123,   751,    89,    90,
     549,   653,   116,  1040,   549,   657,   549,   659,   593,   232,
     115,   677,   760,   761,   119,   115,   114,    42,    44,   119,
      46,    49,   644,   128,   115,   615,   131,   549,   119,   777,
     778,   136,   702,   799,   800,   140,   549,   128,   549,    46,
     131,   549,    76,   807,   696,   136,   712,    44,   710,    46,
      76,     6,   704,   102,   277,    89,    90,    41,    42,    46,
     730,    44,   743,    46,   287,  1102,   677,   116,   117,   735,
      41,   678,    55,   678,    58,    44,   102,   825,   678,   731,
     678,   115,   678,  1120,   307,   119,   743,    58,  1125,    41,
      42,   131,   678,   142,   678,    43,   319,   678,   750,   116,
     678,   712,   709,   857,   709,    44,    58,    46,   678,   709,
     678,   709,   678,   709,   766,   678,   678,    42,    43,   726,
      43,   726,   678,   709,   735,   709,   774,   103,   709,   678,
     726,   709,   355,   678,  1081,   678,   806,   803,   817,   709,
     726,   709,   531,   709,   814,    41,   709,   709,     7,   801,
     831,   749,   541,   709,   961,   817,   678,    42,   810,    42,
     709,    53,    54,    23,   709,   678,   709,   678,   820,   558,
     678,   560,   828,   116,   831,   831,   910,   839,    41,    71,
      72,    58,   767,   839,   774,    45,    41,   709,   411,   136,
     856,    41,   803,   863,    40,    58,   709,   116,   709,    45,
       0,   709,   425,    41,    42,    42,    52,  1154,    58,    43,
     761,    50,    51,    52,   825,    42,   439,    77,    78,    79,
      58,    43,    82,    42,    84,    85,   777,   778,    42,   910,
      55,   985,   136,   137,   138,   842,    42,   842,   825,    85,
      86,    44,   842,    42,   842,   856,   842,    44,   910,    42,
      43,    97,    42,   851,   910,    42,   842,    42,   842,    80,
    1005,   842,    42,  1011,   842,    43,   489,    88,    89,    90,
     116,   102,   842,   943,   842,   103,   842,    41,  1141,   842,
     842,  1144,  1145,  1146,  1034,    42,   842,   935,    89,    90,
     942,    44,   962,   842,   964,  1158,   116,   842,   960,   842,
     116,   963,    73,    74,    75,   971,    42,    43,   154,   979,
      76,   959,   104,   105,   160,   161,    46,   163,    42,     3,
     842,   710,     3,   102,   170,   548,     3,  1081,   998,   842,
      41,   842,     3,   116,   842,   116,    42,   989,    42,     4,
     186,     6,    49,    41,    44,   191,    42,   570,   421,    44,
     961,    44,    46,   426,   991,   201,   202,    46,    41,    47,
     971,    41,    44,    91,  1034,    90,    42,    89,    42,   215,
      78,    76,  1038,  1025,   961,    41,    49,    41,   976,    44,
     226,    46,   980,   229,    42,  1033,   232,   233,   234,   612,
      55,   237,   615,   239,   240,   241,   242,  1049,    41,    41,
    1154,   132,   133,   134,   135,   136,   137,   138,    42,    42,
     799,   800,    43,    42,    42,  1067,  1082,    43,   807,   134,
     135,   136,   137,   138,   647,    42,    41,  1038,    55,    42,
      44,   277,   821,   822,   823,   102,  1088,    42,  1108,   142,
    1106,   287,   288,   289,  1034,    47,   292,  1113,  1114,  1115,
     103,    42,  1050,   102,  1052,    44,    63,    23,    43,    46,
      42,   307,   308,   309,   853,  1117,  1103,  1133,   314,    42,
      42,  1082,    42,  1135,    42,    42,    55,  1129,    42,    45,
      42,   327,   328,   329,    44,    44,    42,   125,   334,  1141,
      47,     3,  1144,  1145,  1146,  1106,    76,    55,    44,    46,
     346,   347,  1113,  1114,  1115,    76,  1158,   730,    47,   355,
      47,    77,    78,    79,    42,    76,    82,   363,    84,    85,
      42,    76,  1133,    42,    42,    42,   156,   157,   158,    41,
     376,    42,    55,    43,   103,    44,    68,    44,    50,    51,
      52,    43,    42,   766,   767,    44,    58,    76,    44,    76,
      76,   774,   118,    44,    44,    76,    76,    44,    90,   405,
     406,   407,   103,    95,     3,   411,   412,   413,    42,    44,
     416,   417,   418,    43,   420,    41,    49,    89,    90,   425,
      47,    44,     3,    44,    41,     6,   975,    41,    44,    42,
      42,    41,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    41,   115,    47,    49,    42,   119,    42,    42,
      49,    50,    51,    52,    42,   245,   128,    42,   532,   131,
      46,    44,     3,   696,   136,     6,   557,   726,   140,  1108,
     565,   704,   478,   727,   264,   138,   266,   267,   268,   269,
     270,   271,   726,   489,   726,    39,   825,   493,   618,   587,
      89,    90,   184,  1072,   452,    50,   188,   189,   731,   851,
      41,   976,   508,   867,   975,   197,   512,     3,    89,    90,
     516,   517,   602,   519,  1117,   992,   115,   750,   210,   177,
     119,  1098,   711,   104,   105,    -1,   218,   219,    -1,   128,
      -1,    -1,   131,   323,   540,    -1,   326,    -1,   544,   231,
     232,   140,   548,   235,    -1,    41,    -1,    43,    89,    90,
      -1,   243,    -1,    49,    -1,    -1,    -1,    -1,    -1,    55,
      -1,    -1,    58,    -1,   570,   571,   572,    -1,   358,    -1,
     576,   577,   578,    -1,   115,    -1,    -1,   810,   119,    -1,
      76,    -1,    -1,    -1,   106,   107,    -1,   128,   110,    -1,
     131,    -1,    -1,    89,    90,   136,    -1,    -1,     3,   140,
      -1,    -1,   110,    -1,     3,    -1,    -1,   103,    -1,   615,
     132,   133,   134,   135,   136,   137,   138,    -1,    -1,   115,
      -1,    -1,    -1,   119,   132,   133,   134,   135,   136,   137,
     138,    -1,   128,    -1,    -1,   131,    41,    -1,    -1,    44,
     136,   647,    41,    -1,   140,    -1,    -1,   653,    -1,    -1,
      49,  1034,    -1,    -1,  1037,  1038,    -1,   663,    -1,   665,
       3,    -1,    -1,   669,   670,   455,   456,   457,   458,   459,
     460,   461,   462,   463,   464,   465,   466,   467,   468,   469,
     470,   471,   472,    -1,    89,    90,    -1,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    -1,   702,    -1,    41,    -1,
     490,    -1,    -1,    -1,    -1,    -1,    49,    -1,   498,   942,
     115,    -1,    55,    -1,   119,    58,   115,    -1,    -1,   725,
     119,    -1,    -1,   128,   730,    -1,   131,   733,    -1,   128,
      -1,   136,   131,    76,    -1,   140,    -1,   743,    -1,   161,
      -1,   140,    -1,    -1,     3,    -1,    89,    90,   170,    -1,
       6,    50,    51,    52,    -1,    -1,   989,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    -1,   774,    -1,
      -1,    -1,   115,    -1,    -1,    -1,   119,    -1,    -1,    -1,
      -1,    80,    41,    -1,    -1,   128,   792,   793,   131,    88,
      89,    90,  1025,   136,    -1,    -1,    55,   140,    -1,    58,
     806,    -1,   592,    -1,     3,    -1,    -1,     3,   814,    -1,
      -1,    -1,   234,    -1,    -1,   237,  1049,    76,   608,   609,
     610,   611,   612,    -1,   614,   831,    -1,    41,    -1,   128,
      89,    90,   131,    -1,  1067,   527,    50,    51,    52,   531,
     532,    55,    41,    -1,    -1,    41,    -1,   539,   540,   541,
      49,    -1,    -1,    49,    -1,  1088,   115,   863,    -1,    58,
     119,    -1,    58,    -1,    -1,   871,    80,    -1,   560,   128,
     562,    -1,   131,    -1,    88,    89,    90,   136,    -1,   571,
      -1,   140,    -1,    -1,  1117,    -1,   308,    -1,    -1,    -1,
      89,    90,   314,    89,    90,    -1,     3,    -1,    -1,    -1,
      -1,    -1,     3,    -1,   910,   327,   328,   329,    -1,    -1,
      -1,    -1,   334,    -1,   128,    -1,   115,   131,    -1,   115,
     119,    -1,    -1,   119,    -1,    -1,    -1,    -1,    -1,   128,
      -1,    -1,   131,    -1,    41,    -1,    -1,   943,    -1,     3,
      41,   140,    49,    -1,   140,    -1,   736,    -1,    49,    -1,
      -1,    58,    -1,   959,    -1,    -1,   962,    58,   964,    -1,
      -1,   751,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   979,   764,   765,   766,    41,   768,    -1,
     986,    -1,    89,    90,    -1,    49,   776,    -1,    89,    90,
     412,   413,   998,    -1,   416,   417,   418,    -1,   420,    -1,
      -1,    41,    -1,    -1,    44,    -1,    -1,    -1,   115,    -1,
     702,    -1,   119,    -1,   115,    -1,    -1,    -1,   119,   711,
      -1,   128,    -1,     3,   131,    89,    90,   128,  1034,    -1,
     131,    -1,  1038,   140,    -1,    -1,    -1,    -1,    -1,   140,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,   115,    -1,    -1,    -1,   119,    -1,  1063,    -1,    -1,
      -1,    41,    -1,    -1,    -1,    -1,    -1,   857,    -1,    49,
      -1,   493,    -1,    -1,    -1,   115,   140,    -1,    58,   119,
      -1,    -1,    -1,    -1,    -1,    -1,   508,    -1,   128,    -1,
     512,   131,  1098,    -1,   516,   517,   136,   519,    -1,    -1,
     140,    -1,  1108,    -1,    -1,    -1,    -1,   799,   800,    89,
      90,    -1,   804,   805,    -1,   807,    -1,    -1,    -1,    -1,
      -1,     3,    -1,    -1,     6,    -1,    -1,    -1,    -1,   919,
     920,    -1,    -1,    -1,    -1,   115,    -1,    -1,   830,   119,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,   128,     3,
     572,   131,     6,     7,   576,   577,   578,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    -1,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    -1,    -1,
      -1,    -1,    46,    -1,    -1,    -1,    50,    51,    52,    53,
      54,    -1,    56,    -1,    -1,    -1,    -1,    89,    90,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,
      -1,    -1,   104,   105,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   653,    86,   115,    -1,    89,    90,   119,    -1,    -1,
      -1,   663,    -1,   665,    -1,    -1,   128,   669,   670,   131,
     104,   105,   944,    -1,   136,    -1,    -1,  1047,    -1,    -1,
      -1,   115,    -1,    -1,     3,   119,    -1,    -1,    -1,   106,
     107,   108,   109,   110,     3,    -1,    -1,   131,    -1,  1069,
     134,   135,   136,    -1,    -1,    -1,   140,   141,    -1,    -1,
      -1,  1081,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,    41,     1,    -1,     3,    -1,    -1,     6,     7,
      49,   733,    41,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    -1,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,    50,    51,    52,    53,    54,    -1,    56,    -1,
      89,    90,    -1,    -1,  1154,    -1,    -1,    -1,    -1,    -1,
     792,   793,    70,    71,    72,    -1,   115,    -1,    -1,    -1,
     119,    -1,    -1,    -1,    -1,    -1,   115,    -1,    86,    -1,
     119,    89,    90,    -1,    -1,    -1,    -1,    -1,    -1,   128,
      -1,   140,   131,    -1,    -1,    -1,    -1,   136,    -1,    -1,
      -1,   140,    -1,    -1,    -1,    -1,    -1,   115,    -1,    -1,
      -1,   119,     1,    -1,     3,     4,    -1,    -1,    -1,     8,
       9,    10,    -1,   131,    -1,    -1,   134,   135,   136,    -1,
      -1,    -1,   140,   141,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    -1,    -1,    -1,    44,    45,    -1,    47,    -1,
      -1,    50,    51,    52,    53,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    77,    78,
      79,    -1,    81,    82,    83,    84,    85,    86,    87,    -1,
      89,    90,    91,    -1,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,    -1,    -1,    -1,    -1,    -1,   106,   107,
     108,   109,   110,    -1,   113,    -1,   115,    -1,    -1,   118,
     119,   120,    -1,    -1,     1,   124,     3,     4,    -1,    -1,
      -1,     8,     9,    10,   132,   133,   134,   135,   136,   137,
     138,   140,    -1,    -1,    -1,    -1,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    -1,    -1,    -1,    44,    45,    -1,
      47,    -1,    -1,    50,    51,    52,    53,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      77,    78,    79,    -1,    81,    82,    83,    84,    85,    86,
      87,    -1,    89,    90,    91,     3,    -1,    -1,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   113,    -1,   115,    -1,
      -1,   118,   119,   120,    -1,    -1,    -1,   124,    -1,    -1,
      -1,    -1,    -1,    41,    -1,    -1,     3,    -1,    -1,     6,
       7,    49,     3,   140,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    -1,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    -1,    -1,    -1,    -1,    46,
      41,    89,    90,    50,    51,    52,    53,    54,    49,    56,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    70,    71,    72,    -1,   115,    -1,    -1,
      -1,   119,    -1,    -1,   106,   107,   108,   109,   110,    86,
      -1,    -1,    89,    90,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,   140,    -1,    -1,    -1,    -1,   104,   105,   131,
     132,   133,   134,   135,   136,   137,   138,    -1,   115,    -1,
      -1,    -1,   119,    -1,   115,    -1,    -1,    -1,   119,    -1,
      -1,    -1,    -1,    -1,   131,    -1,    -1,   134,   135,   136,
      -1,    -1,     3,   140,   141,     6,     7,    -1,     3,   140,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    -1,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    -1,    -1,    -1,    -1,    -1,    41,    -1,    -1,    50,
      51,    52,    53,    54,    49,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    58,     3,    -1,    -1,     6,     7,    70,
      71,    72,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    -1,    86,    -1,    -1,    89,    90,
      -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    -1,
      -1,    -1,    41,   104,   105,    -1,     3,    -1,    -1,    -1,
      49,    -1,    -1,    -1,   115,    -1,    -1,    56,   119,    -1,
     115,    -1,    -1,    -1,   119,    -1,    -1,    -1,    -1,    -1,
     131,    -1,    -1,   134,   135,   136,    -1,    -1,     3,   140,
     141,     6,     7,    -1,    41,   140,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    -1,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    49,    50,    51,    52,    53,    54,
      -1,    56,    89,    90,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   115,    -1,
      -1,    86,   119,    -1,    89,    90,    -1,    -1,    -1,    -1,
      -1,   128,    -1,    -1,   131,    -1,    -1,    -1,    -1,   136,
      -1,    -1,    -1,   140,    -1,    -1,    -1,    -1,    -1,    -1,
     115,    -1,    -1,    -1,   119,    -1,    -1,    -1,    -1,   124,
      -1,    -1,    -1,    -1,    -1,    -1,   131,    -1,    -1,   134,
     135,   136,    -1,    -1,     3,   140,   141,     6,     7,    -1,
       3,    -1,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    -1,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    -1,    -1,    -1,    -1,    -1,    41,    -1,
      -1,    50,    51,    52,    53,    54,    49,    56,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    58,    -1,    -1,    -1,    -1,
      -1,    70,    71,    72,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,
      89,    90,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   104,   105,    -1,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   115,    -1,    -1,    -1,
     119,    -1,   115,    -1,    -1,    -1,   119,    -1,    -1,    -1,
      -1,    -1,   131,    -1,    -1,   134,   135,   136,    -1,    -1,
       3,   140,   141,     6,     7,    -1,    41,   140,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      -1,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    50,    51,    52,
      53,    54,    -1,    56,    89,    90,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     115,    -1,    -1,    86,   119,    -1,    89,    90,    -1,    -1,
      -1,    -1,    -1,   128,    -1,    -1,   131,    -1,    -1,    -1,
      -1,   136,    -1,    -1,    -1,   140,    -1,    -1,    -1,    -1,
      -1,    -1,   115,    -1,    -1,    -1,   119,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   131,    -1,
      -1,   134,   135,   136,    -1,    -1,     3,   140,   141,     6,
       7,    -1,     3,    -1,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    -1,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    -1,    -1,    -1,    -1,    76,
      41,    -1,    -1,    50,    51,    52,    53,    54,    49,    56,
      -1,    -1,    59,    -1,    -1,    -1,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    70,    71,    72,    -1,    -1,    -1,   106,
     107,   108,   109,   110,    -1,   112,    -1,    -1,    -1,    86,
      -1,    -1,    89,    90,    -1,    -1,    -1,    -1,    89,    90,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,    -1,    -1,    -1,    -1,    -1,    -1,   115,    -1,
      -1,    -1,   119,    -1,   115,    -1,    -1,    -1,   119,    -1,
      -1,    -1,    -1,    -1,   131,    -1,    -1,   134,   135,   136,
      -1,    -1,     3,   140,   141,     6,     7,    -1,     3,   140,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    -1,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    -1,    -1,    -1,    -1,    -1,    41,    -1,    -1,    50,
      51,    52,    53,    54,    49,    56,    -1,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,
      71,    72,    -1,    -1,    -1,   106,   107,   108,   109,   110,
      -1,   112,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,
      -1,    -1,    -1,    -1,    89,    90,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,    -1,    -1,
      -1,    -1,    -1,    -1,   115,    -1,    -1,    -1,   119,    -1,
     115,    -1,    -1,    -1,   119,    -1,    -1,    -1,    -1,    -1,
     131,    -1,    -1,   134,   135,   136,    -1,    -1,     3,   140,
     141,     6,     7,    -1,    42,   140,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    -1,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    50,    51,    52,    53,    54,
      -1,    56,    -1,    -1,    59,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    70,    71,    72,   106,   107,
     108,   109,   110,    -1,   112,    -1,    -1,    -1,    -1,    -1,
      -1,    86,    -1,    -1,    89,    90,    -1,    -1,    -1,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,    -1,    24,    25,    26,    27,    28,    29,    30,    31,
     115,    33,    -1,    35,   119,    37,    38,    39,    40,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   131,    -1,    -1,   134,
     135,   136,    -1,    -1,     3,   140,   141,     6,     7,    -1,
      42,    -1,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    -1,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    50,    51,    52,    53,    54,    -1,    56,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    70,    71,    72,   106,   107,   108,   109,   110,    -1,
     112,    -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,
      89,    90,    -1,    -1,    -1,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   115,    -1,    -1,    -1,
     119,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   131,    -1,    -1,   134,   135,   136,    -1,    -1,
       3,   140,   141,     6,     7,    -1,    42,    -1,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      -1,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    50,    51,    52,
      53,    54,    -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,
     106,   107,   108,   109,   110,    -1,   112,    -1,    -1,    -1,
      -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,    -1,
      -1,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   115,    -1,    -1,    -1,   119,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   131,    -1,
      -1,   134,   135,   136,    -1,    -1,     3,   140,   141,     6,
       7,    -1,    42,    -1,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    -1,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    50,    51,    52,    53,    54,    -1,    56,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    70,    71,    72,   106,   107,   108,   109,
     110,    -1,   112,    -1,    -1,    -1,    -1,    -1,    -1,    86,
      -1,    -1,    89,    90,    -1,    -1,    -1,   127,   128,   129,
     130,   131,   132,   133,   134,   135,   136,   137,   138,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   115,    -1,
      -1,    -1,   119,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   131,    -1,    -1,   134,   135,   136,
      -1,    -1,     3,   140,   141,     6,     7,    -1,    -1,    -1,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    -1,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    50,
      51,    52,    53,    54,    -1,    56,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,
      71,    72,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,
      -1,     3,    -1,    -1,    -1,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    -1,     3,   115,    -1,    -1,    -1,   119,    41,
      -1,    50,    51,    52,    53,    54,    -1,    49,     3,    41,
      -1,    -1,    44,   134,   135,   136,    58,    -1,    -1,   140,
     141,    70,    71,    72,    -1,    -1,     3,    -1,    -1,    -1,
      -1,    41,    -1,    -1,    44,    -1,    -1,    86,    87,    -1,
      89,    90,    -1,    -1,    -1,    -1,    41,    89,    90,    -1,
      -1,    -1,    -1,    -1,    49,    -1,    -1,    89,    90,    -1,
      -1,    -1,    -1,    -1,    41,    -1,   115,    -1,    -1,    -1,
     119,   120,    49,   115,    -1,    -1,    -1,   119,    -1,    89,
      90,    -1,    -1,   115,    -1,    -1,   128,   119,    -1,   131,
      -1,   140,    -1,    -1,    89,    90,   128,    -1,   140,   131,
      -1,    -1,    -1,    -1,   136,   115,    -1,    -1,   140,   119,
      -1,    -1,    89,    90,    -1,    -1,    -1,    -1,   128,    -1,
     115,   131,    -1,    -1,   119,    -1,   136,    -1,    -1,    -1,
     140,    -1,    -1,   128,    -1,    -1,   131,    -1,   115,    -1,
      -1,    -1,   119,     3,    -1,   140,    -1,    -1,    -1,    -1,
      -1,   128,    -1,    -1,   131,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   140,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,
      50,    51,    52,    53,    54,    -1,    -1,    -1,    58,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,    -1,     3,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,    89,
      90,    -1,    -1,    -1,    -1,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    -1,    -1,    -1,   115,     3,    -1,    -1,   119,
      -1,    -1,    -1,    -1,   124,    -1,    -1,    42,   128,    -1,
      -1,   131,    -1,    -1,    -1,    -1,   136,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    -1,    -1,    86,    -1,    -1,
      89,    90,    49,    50,    51,    52,    53,    54,    -1,    -1,
      -1,    58,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    70,    71,    72,   115,    -1,   103,    -1,
     119,   106,   107,   108,   109,   110,    -1,   112,    -1,    86,
      -1,    -1,    89,    90,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,    -1,    -1,    -1,    -1,   115,     3,
      -1,    -1,   119,    -1,    -1,    -1,    -1,   124,    -1,    -1,
      -1,   128,    -1,    -1,   131,    -1,    -1,    -1,    -1,   136,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    49,    50,    51,    52,    53,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    -1,    86,    -1,    -1,    89,    90,    -1,    -1,    -1,
      -1,    -1,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
      -1,   115,    -1,    -1,    -1,   119,    -1,    -1,    -1,    -1,
     124,    53,    54,    -1,   128,    -1,    -1,   131,    -1,    -1,
      -1,    -1,   136,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,   106,   107,   108,   109,   110,    -1,   112,    -1,    -1,
      -1,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,
      -1,    -1,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   115,    -1,     0,     1,   119,     3,     4,
      -1,    -1,    -1,     8,     9,    10,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   136,    -1,    -1,    -1,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    -1,    -1,    -1,    44,
      45,    -1,    -1,    -1,    -1,    50,    51,    52,    53,    54,
      -1,    -1,    57,    -1,    -1,    -1,    -1,    -1,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    -1,    -1,
      -1,    -1,    77,    78,    79,    -1,    81,    82,    83,    84,
      85,    86,    87,    -1,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   106,   107,   108,
     109,   110,    -1,    -1,    -1,    -1,    -1,    -1,   113,     1,
     115,     3,     4,   118,   119,    -1,     8,     9,    10,   124,
     125,   130,   131,   132,   133,   134,   135,   136,   137,   138,
      -1,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
      -1,    -1,    44,    45,    -1,    47,    -1,    -1,    50,    51,
      52,    53,    54,    -1,    -1,    57,    -1,    -1,    -1,    -1,
      -1,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    -1,    -1,    -1,    -1,    77,    78,    79,    -1,    81,
      82,    83,    84,    85,    86,    87,    -1,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   113,     1,   115,     3,     4,   118,   119,    -1,     8,
       9,    10,   124,   125,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    -1,    -1,    -1,    44,    45,    -1,    47,    -1,
      -1,    50,    51,    52,    53,    54,    -1,    -1,    57,    -1,
      -1,    -1,    -1,    -1,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    -1,    -1,    -1,    -1,    77,    78,
      79,    -1,    81,    82,    83,    84,    85,    86,    87,    -1,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   113,     1,   115,     3,     4,   118,
     119,    -1,     8,     9,    10,   124,   125,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    -1,    -1,    -1,    44,    45,
      -1,    47,    -1,    -1,    50,    51,    52,    53,    54,    -1,
      -1,    57,    -1,    -1,    -1,    -1,    -1,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    -1,    -1,    -1,
      -1,    77,    78,    79,    -1,    81,    82,    83,    84,    85,
      86,    87,    -1,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   113,     1,   115,
       3,     4,   118,   119,    -1,     8,     9,    10,   124,   125,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    -1,    -1,
      -1,    44,    45,    -1,    -1,    -1,    -1,    50,    51,    52,
      53,    54,    -1,    -1,    57,    -1,    -1,    -1,    61,    -1,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      -1,    -1,    -1,    -1,    77,    78,    79,    -1,    81,    82,
      83,    84,    85,    86,    87,    -1,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    -1,
       1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     113,    -1,   115,    -1,    -1,   118,   119,    -1,    -1,    -1,
      -1,   124,   125,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,
      51,    52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,
      71,    72,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   115,    -1,    -1,    -1,   119,    -1,
      -1,    -1,    -1,   124,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,
      50,    51,    52,    53,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,    89,
      90,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   115,    -1,    -1,    -1,   119,
      -1,    -1,    -1,    -1,   124,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      49,    50,    51,    52,    53,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    70,    71,    72,   106,   107,   108,   109,   110,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,
      89,    90,    -1,    -1,    -1,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,     3,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   115,    13,    -1,    -1,
     119,    -1,    -1,    -1,    -1,   124,   125,    -1,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    49,    50,    51,    52,    53,    54,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      86,    -1,    -1,    89,    90,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   115,
      -1,    -1,    -1,   119,    -1,    -1,    -1,    -1,   124,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    49,    50,    51,    52,    53,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    86,    -1,    -1,    89,    90,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     115,    -1,    -1,   118,   119,    -1,    -1,    -1,    -1,   124,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    49,    50,    51,    52,    53,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    86,    -1,    -1,    89,    90,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
       3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   115,    -1,    -1,    -1,   119,    -1,    -1,    -1,    -1,
     124,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    50,    51,    52,
      53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,
       3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,    -1,
      -1,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    -1,    -1,
      -1,    -1,   115,    -1,    -1,    -1,   119,    50,    51,    52,
      53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,
     106,   107,   108,   109,   110,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    86,    87,    -1,    89,    90,    -1,    -1,
      -1,    -1,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       3,    -1,   115,     6,     7,    -1,   119,   120,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      -1,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    50,    51,    52,
      53,    54,    -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,    -1,
      -1,    -1,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
      -1,    -1,   115,    -1,    -1,    -1,   119,    49,    50,    51,
      52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,
      -1,    -1,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      -1,    -1,    -1,   115,    -1,    -1,    -1,   119,    50,    51,
      52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,
      -1,    -1,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
      -1,    -1,    -1,   115,    -1,    -1,    -1,   119,    50,    51,
      52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    86,    87,    -1,    89,    90,    -1,
      -1,    -1,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
      -1,    -1,    -1,   115,    -1,    -1,    -1,   119,    50,    51,
      52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,
      -1,    -1,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
      -1,    -1,    -1,   115,    -1,    -1,    -1,   119,    -1,    -1,
      -1,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,
      -1,    -1,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
      -1,    -1,    -1,   115,    -1,    -1,    -1,   119,    -1,    -1,
      -1,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,
      72,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    86,    -1,    -1,    89,    90,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,
     102,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   115,    -1,    -1,    -1,   119,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    53,    54,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    71,    72,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      86,    -1,    -1,    89,    90,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     3,    -1,    -1,   102,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   115,
      -1,    -1,    -1,   119,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    53,    54,    -1,    -1,     4,    -1,    -1,
      -1,    -1,    -1,    10,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    71,    72,    -1,    -1,    -1,    23,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,    89,
      90,    -1,    -1,    -1,    -1,    -1,    -1,    44,    45,    -1,
      47,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   115,    63,    64,    65,   119,
      67,    68,    69,     4,    -1,    -1,    73,    74,    75,    10,
      77,    78,    79,    -1,    81,    82,    83,    84,    85,    -1,
      -1,    -1,    23,    -1,    91,    -1,    -1,    -1,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    44,    45,    -1,   113,    -1,    -1,    -1,
      -1,   118,    -1,    -1,    59,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    63,    64,    65,    -1,    67,    68,    69,    -1,
      -1,    -1,    73,    74,    75,    -1,    77,    78,    79,    -1,
      81,    82,    83,    84,    85,    -1,    -1,    59,    -1,    -1,
      91,    -1,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   106,   107,   108,   109,   110,    -1,   112,    -1,    -1,
      -1,    -1,   113,    -1,    -1,    -1,    -1,   118,    -1,    -1,
      59,    -1,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   106,   107,   108,   109,   110,    -1,
     112,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    59,    -1,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   106,   107,   108,
     109,   110,    -1,   112,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    59,    -1,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     106,   107,   108,   109,   110,    -1,   112,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   106,   107,   108,   109,   110,    -1,   112,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int16 yystos[] =
{
       0,   121,   122,   123,   144,   145,     1,     3,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    49,    50,    51,    52,    53,
      54,    70,    71,    72,    86,    89,    90,   115,   119,   124,
     194,   235,   236,   254,   255,   257,   258,   259,   262,   263,
     264,   292,   293,   308,   311,   313,     1,   236,     1,    41,
       0,     1,     4,     8,     9,    10,    23,    44,    45,    57,
      63,    64,    65,    66,    67,    68,    69,    77,    78,    79,
      81,    82,    83,    84,    85,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   113,   118,   124,   125,
     146,   147,   148,   150,   151,   152,   153,   156,   157,   159,
     160,   161,   162,   163,   164,   167,   168,   169,   172,   174,
     179,   180,   181,   182,   184,   188,   196,   197,   198,   199,
     200,   204,   205,   211,   212,   223,   224,   230,   231,   232,
      53,    71,    49,   283,   283,    41,   142,   102,   307,   235,
     311,     0,   258,   254,    41,    49,    55,    58,    76,   119,
     128,   131,   136,   140,   243,   244,   246,   248,   249,   250,
     251,   311,   254,   263,   311,   307,   117,   142,   312,     0,
     233,   234,   236,   120,    41,     6,   118,   317,    41,   320,
       1,   256,   257,   308,   320,    41,   166,    41,    41,    41,
      41,    72,    92,   311,    46,   308,   311,    41,     4,    46,
      41,     4,   317,    41,   178,   256,   176,   178,    41,    41,
     317,    41,    32,    34,    36,   102,   187,   259,   262,   293,
     311,   320,    41,   128,   131,   246,   248,   251,   311,    87,
     194,   256,   293,    45,   232,   260,   118,   119,   313,   314,
       3,     7,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    41,    56,   131,   134,   135,   136,
     140,   141,   236,   237,   238,   240,   256,   257,   277,   278,
     279,   280,   281,   282,   317,   318,   254,    41,   128,   131,
     233,   249,   251,   311,    46,   104,   105,   265,   266,   267,
     268,   277,   280,    59,   277,   277,     3,    41,    49,   140,
     247,   250,   283,   311,    49,   247,   250,   283,   251,   254,
     311,   243,    41,    58,   243,    41,    58,    49,   128,   131,
     247,   250,   283,   311,   116,   313,   119,   314,    42,    43,
     124,   125,   266,   308,   309,   317,   187,   293,     6,    46,
     309,   321,   309,     0,    44,    41,   246,   248,    55,    53,
      54,    71,   294,   295,   308,   308,   309,    13,   173,   233,
     233,   311,   311,    44,    55,   214,    55,    46,   308,   175,
     321,   233,    46,   245,   246,   248,   249,    44,    43,   177,
     309,   310,   154,   155,   317,   233,   118,   208,   209,   236,
     292,   311,   311,   131,   317,   128,   131,   251,   308,   311,
     321,    41,    49,    49,   309,    41,    49,   128,   131,   311,
     116,   246,   311,   269,   308,    41,   246,    76,   284,   285,
     311,   317,     1,   261,   277,   308,   312,    48,   111,   277,
     279,     3,    41,    49,   279,   277,   277,   277,   277,   277,
     277,   103,    43,   239,    41,   106,   107,   108,   109,   110,
     112,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,    48,   111,     7,   128,   131,   251,   311,
     248,   311,   248,    42,    42,   128,   131,   248,   311,   116,
      58,    59,    41,   251,   311,   247,   311,    41,    58,   247,
     251,   233,    59,   277,   233,    59,   277,   247,    49,   247,
     250,   283,    49,   247,   250,   283,   116,    49,   128,   131,
     247,   254,   312,     0,   125,   233,    43,    42,   311,   311,
     183,    43,    55,    42,   243,    41,   265,   311,    42,    55,
      42,    43,   171,    43,    42,    42,    44,    44,   256,   145,
     311,   213,    42,    42,    42,   176,    41,   178,    42,    42,
      43,    46,    42,   102,   103,    43,   210,   243,    41,    60,
     116,    42,   251,   311,    44,   233,    49,    49,   116,    80,
      88,    89,    90,   128,   131,   252,   253,   254,   297,   300,
     301,   302,    55,    76,   195,   233,   311,   302,   286,    46,
      44,    90,   299,    42,    42,   307,     3,     3,    42,   103,
     128,   131,   136,   251,   102,   256,   279,    41,   240,   277,
     277,   277,   277,   277,   277,   277,   277,   277,   277,   277,
     277,   277,   277,   277,   277,   277,   277,     3,     3,   311,
     116,    42,    42,    42,   116,   246,   249,   254,   277,   233,
     247,   311,    42,   116,   233,    59,   277,    42,    59,    42,
      59,   247,   247,    49,   128,   131,   247,   250,   247,    49,
      49,   247,   233,   233,     4,    46,   317,    41,   145,   321,
     154,   281,   317,   322,    44,   233,    44,    46,     4,   165,
     317,     4,    44,    46,   114,   170,   246,   317,   319,   309,
     317,   322,    42,   236,   246,    46,   245,    47,    44,   145,
     230,   176,    41,    47,   233,   177,   115,   119,   308,   315,
      44,   321,   236,   170,   208,    91,   206,   209,   233,   158,
     254,   246,   317,   116,    42,    41,    41,   297,    90,    89,
     300,   253,    46,   111,    58,   189,   265,   258,    44,    46,
      42,    55,   186,   243,    78,   287,   288,   296,   202,   284,
      41,    41,   277,   277,    42,    42,   136,   254,    42,   128,
     131,   277,   244,    42,   256,   239,    76,    41,    41,    42,
      42,   246,   249,    59,    42,    42,   247,    42,    59,   252,
     252,   247,    49,    49,   247,   247,   247,   233,    47,    43,
      43,    42,   149,    41,   302,    55,    42,    43,   171,   170,
     246,   302,    44,    47,   256,   308,    44,    55,   319,   233,
      42,   142,   117,   142,   316,   102,    42,    47,   103,   311,
      45,   118,   184,   200,   204,   205,   207,   220,   222,   230,
     210,    42,   145,   302,    44,   233,   277,   187,   102,   190,
      44,   152,   270,   271,   272,   273,    41,    55,   300,   302,
     303,     1,     1,    43,    44,    46,   185,    43,    73,    74,
      75,   289,   291,     1,    44,    66,    73,    74,    75,   124,
     150,   151,   152,   153,   156,   157,   161,   164,   167,   169,
     172,   174,   179,   180,   181,   182,   200,   204,   205,   211,
     215,   218,   219,   220,   221,   222,   223,   225,   226,   228,
     230,    46,   241,   280,   241,    42,   277,   277,   277,    42,
      42,    42,    42,   244,   277,   241,   241,    42,    42,    42,
     247,   247,    42,   321,   321,   252,   215,   233,   170,   317,
     322,    44,   246,    42,   302,    44,   246,    44,   178,    42,
     252,   119,   308,   308,   119,   308,   237,     4,    46,   293,
      55,   102,    87,   120,   256,    61,    44,    42,    42,   297,
     298,    41,    46,   191,   270,   124,   274,   275,   308,    47,
      43,   124,   125,   233,   265,    55,    76,   304,    44,   246,
     288,   311,   290,    46,    76,    76,    76,   218,    47,   125,
     124,   215,    32,    34,    36,   140,   229,   262,   311,   201,
      42,    43,   242,    42,   277,   277,    42,    42,    42,   243,
      47,    42,   171,   302,    44,   246,   170,    44,    44,   316,
     316,   103,   243,   311,   256,   208,   311,   256,    41,   297,
     186,   233,    41,    44,   192,   275,   271,    55,    44,   246,
     125,   272,    43,    42,    44,   266,   305,   306,   311,    44,
      46,   302,   283,   296,   216,    44,   203,   246,   313,    76,
      44,   215,   280,    44,    44,    44,   302,    44,   243,   245,
     103,    55,    41,   128,   131,   251,   233,   185,    42,   193,
     276,   277,   302,   272,   272,    44,    46,    44,    43,    49,
      41,    46,   186,   311,   215,   302,    41,   277,    47,   242,
      44,    44,   233,    41,    41,    41,   131,    42,   302,    44,
     186,   306,   185,   283,    47,   186,   233,    44,   203,    42,
     233,   233,   233,    41,   303,   111,   191,   185,    49,   217,
     185,    42,   227,   302,    42,    42,    42,   233,   256,   192,
     215,   227,    44,    46,    55,   227,   227,   227,    42,   191,
     265,   227,   192,    44
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int16 yyr1[] =
{
       0,   143,   144,   144,   144,   144,   144,   144,   144,   145,
     145,   145,   145,   146,   146,   146,   146,   146,   146,   146,
     147,   147,   147,   147,   147,   147,   147,   147,   147,   147,
     147,   147,   147,   147,   147,   147,   147,   147,   147,   149,
     148,   150,   151,   152,   152,   152,   152,   152,   153,   153,
     154,   155,   155,   156,   156,   156,   158,   157,   159,   159,
     160,   160,   161,   161,   161,   161,   162,   163,   163,   164,
     164,   165,   165,   166,   166,   167,   167,   167,   168,   168,
     169,   169,   169,   169,   169,   169,   169,   169,   170,   170,
     170,   171,   171,   172,   173,   173,   174,   174,   174,   175,
     176,   177,   177,   178,   178,   178,   179,   180,   181,   182,
     182,   182,   183,   182,   182,   182,   182,   182,   184,   184,
     184,   184,   184,   185,   185,   185,   185,   186,   187,   187,
     187,   187,   187,   187,   187,   188,   188,   188,   189,   190,
     190,   191,   192,   193,   192,   194,   194,   194,   195,   195,
     196,   197,   197,   198,   199,   199,   199,   199,   199,   199,
     201,   200,   202,   200,   203,   203,   204,   206,   205,   205,
     205,   205,   205,   207,   207,   207,   207,   207,   207,   208,
     208,   209,   209,   209,   209,   210,   210,   211,   211,   211,
     213,   212,   214,   212,   212,   215,   215,   216,   217,   215,
     215,   215,   215,   218,   218,   218,   218,   218,   218,   218,
     218,   218,   218,   218,   218,   218,   218,   218,   218,   218,
     218,   219,   219,   219,   220,   221,   222,   222,   222,   222,
     222,   223,   224,   225,   225,   225,   226,   226,   226,   226,
     226,   226,   226,   226,   226,   226,   227,   227,   227,   228,
     229,   229,   229,   229,   229,   230,   230,   231,   231,   232,
     232,   232,   232,   232,   232,   232,   232,   232,   233,   234,
     234,   234,   234,   234,   234,   234,   235,   235,   236,   236,
     237,   238,   238,   239,   239,   240,   240,   241,   241,   242,
     242,   243,   243,   243,   243,   243,   244,   244,   244,   244,
     245,   245,   245,   245,   246,   246,   246,   246,   246,   246,
     246,   246,   246,   246,   246,   246,   246,   246,   246,   246,
     246,   246,   246,   246,   246,   247,   247,   247,   247,   247,
     247,   247,   247,   248,   248,   248,   248,   248,   248,   248,
     248,   248,   248,   248,   248,   248,   249,   249,   249,   249,
     249,   249,   249,   249,   249,   249,   249,   249,   249,   249,
     249,   250,   250,   250,   250,   250,   250,   250,   250,   251,
     251,   251,   251,   252,   252,   252,   253,   253,   254,   254,
     255,   255,   255,   256,   257,   257,   257,   257,   257,   258,
     258,   258,   258,   258,   258,   258,   258,   260,   259,   261,
     261,   262,   263,   263,   264,   264,   264,   264,   264,   264,
     264,   264,   264,   264,   264,   264,   264,   264,   265,   265,
     266,   266,   267,   268,   269,   269,   270,   270,   271,   271,
     272,   272,   272,   272,   272,   272,   272,   273,   274,   274,
     275,   275,   276,   277,   277,   278,   278,   278,   278,   278,
     278,   278,   278,   279,   279,   279,   279,   279,   279,   279,
     279,   279,   279,   280,   280,   280,   280,   280,   280,   280,
     280,   280,   280,   280,   281,   281,   281,   281,   281,   281,
     281,   281,   281,   281,   282,   282,   282,   282,   282,   282,
     282,   282,   282,   282,   282,   282,   282,   282,   282,   282,
     282,   282,   282,   282,   282,   282,   282,   282,   282,   283,
     283,   284,   286,   285,   285,   287,   287,   289,   288,   290,
     288,   291,   291,   291,   292,   292,   293,   293,   293,   294,
     294,   294,   295,   295,   296,   296,   297,   297,   297,   297,
     298,   298,   299,   299,   300,   300,   300,   300,   300,   300,
     301,   301,   301,   302,   302,   303,   303,   303,   303,   303,
     303,   304,   304,   305,   305,   305,   305,   306,   306,   307,
     308,   308,   308,   309,   309,   309,   310,   310,   311,   311,
     311,   311,   311,   311,   311,   312,   312,   312,   312,   313,
     313,   314,   314,   315,   315,   315,   315,   315,   315,   316,
     316,   316,   316,   317,   317,   318,   318,   319,   319,   319,
     320,   320,   321,   321,   321,   321,   321,   321,   322,   322
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     3,     2,     3,     2,     5,     2,     2,
       2,     2,     0,     1,     1,     1,     1,     1,     1,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     0,
       8,     5,     3,     5,     5,     9,     3,     3,     2,     2,
       4,     1,     1,     7,     7,     5,     0,     7,     1,     1,
       2,     2,     1,     5,     5,     5,     3,     7,     8,     5,
       3,     1,     1,     3,     0,     4,     7,     6,     1,     1,
       7,     9,     8,    10,     5,     7,     6,     8,     1,     1,
       5,     4,     5,     7,     1,     3,     6,     6,     8,     1,
       2,     3,     0,     2,     3,     6,     5,     9,     2,     1,
       1,     1,     0,     6,     1,     6,    10,     1,     6,     7,
       9,     5,     6,     1,     5,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     1,    12,    14,     8,     1,     1,
       0,     1,     1,     0,     3,     1,     2,     2,     2,     0,
       5,     8,    11,     6,     1,     1,     1,     1,     1,     1,
       0,    10,     0,     8,     1,     4,     4,     0,     6,     3,
       6,     4,     7,     1,     1,     1,     1,     1,     1,     2,
       0,     2,     7,     6,     1,     3,     0,     3,     4,     4,
       0,     6,     0,     5,     5,     2,     2,     0,     0,     7,
       1,     0,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     2,     6,     7,     8,     8,     8,     9,
       7,     5,     2,     2,     2,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     4,     2,     5,
       1,     1,     1,     1,     1,     1,     0,     1,     2,     1,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     2,     3,     4,     4,     0,     2,     1,     1,     2,
       1,     2,     0,     3,     0,     1,     1,     2,     0,     3,
       0,     2,     5,     2,     2,     0,     2,     2,     1,     5,
       1,     1,     5,     0,     2,     3,     3,     1,     2,     2,
       3,     4,     5,     4,     3,     4,     4,     3,     3,     4,
       5,     6,     6,     5,     5,     1,     2,     3,     4,     5,
       3,     4,     4,     1,     2,     4,     4,     4,     5,     6,
       5,     6,     3,     4,     4,     5,     2,     2,     3,     3,
       3,     3,     1,     2,     2,     2,     2,     2,     3,     3,
       4,     3,     4,     2,     3,     3,     4,     5,     3,     3,
       2,     2,     1,     1,     2,     1,     1,     1,     1,     2,
       1,     1,     1,     1,     2,     1,     2,     3,     2,     1,
       1,     1,     2,     1,     1,     2,     1,     0,     4,     2,
       2,     1,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     0,     1,     2,     1,     0,
       1,     2,     2,     3,     4,     4,     1,     3,     1,     2,
       1,     3,     1,     1,     1,     3,     6,     3,     6,     3,
       6,     3,     6,     1,     1,     1,     5,     6,     4,     2,
       1,     1,     1,     1,     1,     3,     4,     5,     5,     5,
       6,     6,     2,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     5,     5,
       3,     3,     3,     5,     2,     2,     2,     2,     2,     1,
       0,     1,     0,     3,     0,     1,     3,     0,     4,     0,
       6,     1,     1,     1,     2,     2,     1,     1,     1,     1,
       1,     1,     1,     0,     1,     0,     1,     1,     2,     2,
       1,     0,     1,     0,     4,     1,     1,     5,     2,     4,
       1,     1,     2,     1,     0,     3,     3,     4,     4,     3,
       4,     2,     0,     1,     3,     2,     4,     2,     2,     3,
       1,     1,     1,     1,     1,     1,     1,     0,     2,     4,
       1,     3,     1,     2,     3,     3,     2,     2,     2,     1,
       2,     1,     3,     2,     4,     1,     3,     1,     3,     3,
       2,     2,     2,     2,     1,     2,     1,     1,     1,     1,
       3,     0,     3,     5,     1,     3,     3,     5,     1,     1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= END)
    {
      yychar = END;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* program: interface  */
#line 1818 "./CParse/parser.y"
                            {
                   if (!classes) classes = NewHash();
		   Setattr((yyvsp[0].node),"classes",classes); 
		   Setattr((yyvsp[0].node),"name",ModuleName);
		   
		   if ((!module_node) && ModuleName) {
		     module_node = new_node("module");
		     Setattr(module_node,"name",ModuleName);
		   }
		   Setattr((yyvsp[0].node),"module",module_node);
	           top = (yyvsp[0].node);
               }
#line 4966 "CParse/parser.c"
    break;

  case 3: /* program: PARSETYPE parm END  */
#line 1830 "./CParse/parser.y"
                                    {
                 top = Copy(Getattr((yyvsp[-1].p),"type"));
		 Delete((yyvsp[-1].p));
               }
#line 4975 "CParse/parser.c"
    break;

  case 4: /* program: PARSETYPE error  */
#line 1834 "./CParse/parser.y"
                                 {
                 top = 0;
               }
#line 4983 "CParse/parser.c"
    break;

  case 5: /* program: PARSEPARM parm END  */
#line 1837 "./CParse/parser.y"
                                    {
                 top = (yyvsp[-1].p);
               }
#line 4991 "CParse/parser.c"
    break;

  case 6: /* program: PARSEPARM error  */
#line 1840 "./CParse/parser.y"
                                 {
                 top = 0;
               }
#line 4999 "CParse/parser.c"
    break;

  case 7: /* program: PARSEPARMS LPAREN parms RPAREN END  */
#line 1843 "./CParse/parser.y"
                                                    {
                 top = (yyvsp[-2].pl);
               }
#line 5007 "CParse/parser.c"
    break;

  case 8: /* program: PARSEPARMS error  */
#line 1846 "./CParse/parser.y"
                                  {
                 top = 0;
               }
#line 5015 "CParse/parser.c"
    break;

  case 9: /* interface: interface declaration  */
#line 1851 "./CParse/parser.y"
                                       {  
                   /* add declaration to end of linked list (the declaration isn't always a single declaration, sometimes it is a linked list itself) */
                   if (currentDeclComment != NULL) {
		     set_comment((yyvsp[0].node), currentDeclComment);
		     currentDeclComment = NULL;
                   }                                      
                   appendChild((yyvsp[-1].node),(yyvsp[0].node));
                   (yyval.node) = (yyvsp[-1].node);
               }
#line 5029 "CParse/parser.c"
    break;

  case 10: /* interface: interface DOXYGENSTRING  */
#line 1860 "./CParse/parser.y"
                                         {
                   currentDeclComment = (yyvsp[0].str); 
                   (yyval.node) = (yyvsp[-1].node);
               }
#line 5038 "CParse/parser.c"
    break;

  case 11: /* interface: interface DOXYGENPOSTSTRING  */
#line 1864 "./CParse/parser.y"
                                             {
                   Node *node = lastChild((yyvsp[-1].node));
                   if (node) {
                     set_comment(node, (yyvsp[0].str));
                   }
                   (yyval.node) = (yyvsp[-1].node);
               }
#line 5050 "CParse/parser.c"
    break;

  case 12: /* interface: %empty  */
#line 1871 "./CParse/parser.y"
                        {
                   (yyval.node) = new_node("top");
               }
#line 5058 "CParse/parser.c"
    break;

  case 16: /* declaration: SEMI  */
#line 1879 "./CParse/parser.y"
                      { (yyval.node) = 0; }
#line 5064 "CParse/parser.c"
    break;

  case 17: /* declaration: error  */
#line 1880 "./CParse/parser.y"
                       {
		  if (cparse_unknown_directive) {
		      Swig_error(cparse_file, cparse_line, "Unknown directive '%s'.\n", cparse_unknown_directive);
		  } else {
		      Swig_error(cparse_file, cparse_line, "Syntax error in input(1).\n");
		  }
		  Exit(EXIT_FAILURE);
               }
#line 5077 "CParse/parser.c"
    break;

  case 18: /* declaration: c_constructor_decl  */
#line 1889 "./CParse/parser.y"
                                    { 
                  if ((yyval.node)) {
   		      add_symbols((yyval.node));
                  }
                  (yyval.node) = (yyvsp[0].node); 
	       }
#line 5088 "CParse/parser.c"
    break;

  case 19: /* declaration: error CONVERSIONOPERATOR  */
#line 1905 "./CParse/parser.y"
                                          {
                  (yyval.node) = 0;
                  skip_decl();
               }
#line 5097 "CParse/parser.c"
    break;

  case 39: /* $@1: %empty  */
#line 1940 "./CParse/parser.y"
                                                             {
               Node *cls;
	       String *clsname;
	       extendmode = 1;
	       cplus_mode = CPLUS_PUBLIC;
	       if (!classes) classes = NewHash();
	       if (!classes_typedefs) classes_typedefs = NewHash();
	       clsname = make_class_name((yyvsp[-1].str));
	       cls = Getattr(classes,clsname);
	       if (!cls) {
	         cls = Getattr(classes_typedefs, clsname);
		 if (!cls) {
		   /* No previous definition. Create a new scope */
		   Node *am = Getattr(Swig_extend_hash(),clsname);
		   if (!am) {
		     Swig_symbol_newscope();
		     Swig_symbol_setscopename((yyvsp[-1].str));
		     prev_symtab = 0;
		   } else {
		     prev_symtab = Swig_symbol_setscope(Getattr(am,"symtab"));
		   }
		   current_class = 0;
		 } else {
		   /* Previous typedef class definition.  Use its symbol table.
		      Deprecated, just the real name should be used. 
		      Note that %extend before the class typedef never worked, only %extend after the class typedef. */
		   prev_symtab = Swig_symbol_setscope(Getattr(cls, "symtab"));
		   current_class = cls;
		   SWIG_WARN_NODE_BEGIN(cls);
		   Swig_warning(WARN_PARSE_EXTEND_NAME, cparse_file, cparse_line, "Deprecated %%extend name used - the %s name '%s' should be used instead of the typedef name '%s'.\n", Getattr(cls, "kind"), SwigType_namestr(Getattr(cls, "name")), (yyvsp[-1].str));
		   SWIG_WARN_NODE_END(cls);
		 }
	       } else {
		 /* Previous class definition.  Use its symbol table */
		 prev_symtab = Swig_symbol_setscope(Getattr(cls,"symtab"));
		 current_class = cls;
	       }
	       Classprefix = NewString((yyvsp[-1].str));
	       Namespaceprefix= Swig_symbol_qualifiedscopename(0);
	       Delete(clsname);
	     }
#line 5143 "CParse/parser.c"
    break;

  case 40: /* extend_directive: EXTEND options classkeyopt idcolon LBRACE $@1 cpp_members RBRACE  */
#line 1980 "./CParse/parser.y"
                                  {
               String *clsname;
	       extendmode = 0;
               (yyval.node) = new_node("extend");
	       Setattr((yyval.node),"symtab",Swig_symbol_popscope());
	       if (prev_symtab) {
		 Swig_symbol_setscope(prev_symtab);
	       }
	       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
               clsname = make_class_name((yyvsp[-4].str));
	       Setattr((yyval.node),"name",clsname);

	       mark_nodes_as_extend((yyvsp[-1].node));
	       if (current_class) {
		 /* We add the extension to the previously defined class */
		 appendChild((yyval.node), (yyvsp[-1].node));
		 appendChild(current_class,(yyval.node));
	       } else {
		 /* We store the extensions in the extensions hash */
		 Node *am = Getattr(Swig_extend_hash(),clsname);
		 if (am) {
		   /* Append the members to the previous extend methods */
		   appendChild(am, (yyvsp[-1].node));
		 } else {
		   appendChild((yyval.node), (yyvsp[-1].node));
		   Setattr(Swig_extend_hash(),clsname,(yyval.node));
		 }
	       }
	       current_class = 0;
	       Delete(Classprefix);
	       Delete(clsname);
	       Classprefix = 0;
	       prev_symtab = 0;
	       (yyval.node) = 0;

	     }
#line 5184 "CParse/parser.c"
    break;

  case 41: /* apply_directive: APPLY typemap_parm LBRACE tm_list RBRACE  */
#line 2022 "./CParse/parser.y"
                                                           {
                    (yyval.node) = new_node("apply");
                    Setattr((yyval.node),"pattern",Getattr((yyvsp[-3].p),"pattern"));
		    appendChild((yyval.node),(yyvsp[-1].p));
               }
#line 5194 "CParse/parser.c"
    break;

  case 42: /* clear_directive: CLEAR tm_list SEMI  */
#line 2032 "./CParse/parser.y"
                                     {
		 (yyval.node) = new_node("clear");
		 appendChild((yyval.node),(yyvsp[-1].p));
               }
#line 5203 "CParse/parser.c"
    break;

  case 43: /* constant_directive: CONSTANT identifier EQUAL definetype SEMI  */
#line 2046 "./CParse/parser.y"
                                                                {
		 SwigType *type = NewSwigType((yyvsp[-1].dtype).type);
		 if (Len(type) > 0) {
		   (yyval.node) = new_node("constant");
		   Setattr((yyval.node), "name", (yyvsp[-3].id));
		   Setattr((yyval.node), "type", type);
		   Setattr((yyval.node), "value", (yyvsp[-1].dtype).val);
		   if ((yyvsp[-1].dtype).rawval) Setattr((yyval.node), "rawval", (yyvsp[-1].dtype).rawval);
		   Setattr((yyval.node), "storage", "%constant");
		   SetFlag((yyval.node), "feature:immutable");
		   add_symbols((yyval.node));
		   Delete(type);
		 } else {
		   Swig_warning(WARN_PARSE_UNSUPPORTED_VALUE, cparse_file, cparse_line, "Unsupported constant value (ignored)\n");
		   (yyval.node) = 0;
		 }
	       }
#line 5225 "CParse/parser.c"
    break;

  case 44: /* constant_directive: CONSTANT type declarator def_args SEMI  */
#line 2063 "./CParse/parser.y"
                                                        {
		 SwigType_push((yyvsp[-3].type), (yyvsp[-2].decl).type);
		 /* Sneaky callback function trick */
		 if (SwigType_isfunction((yyvsp[-3].type))) {
		   SwigType_add_pointer((yyvsp[-3].type));
		 }
		 (yyval.node) = new_node("constant");
		 Setattr((yyval.node), "name", (yyvsp[-2].decl).id);
		 Setattr((yyval.node), "type", (yyvsp[-3].type));
		 Setattr((yyval.node), "value", (yyvsp[-1].dtype).val);
		 if ((yyvsp[-1].dtype).rawval) Setattr((yyval.node), "rawval", (yyvsp[-1].dtype).rawval);
		 Setattr((yyval.node), "storage", "%constant");
		 SetFlag((yyval.node), "feature:immutable");
		 add_symbols((yyval.node));
               }
#line 5245 "CParse/parser.c"
    break;

  case 45: /* constant_directive: CONSTANT type direct_declarator LPAREN parms RPAREN cv_ref_qualifier def_args SEMI  */
#line 2080 "./CParse/parser.y"
                                                                                                    {
		 SwigType_add_function((yyvsp[-7].type), (yyvsp[-4].pl));
		 SwigType_push((yyvsp[-7].type), (yyvsp[-2].dtype).qualifier);
		 SwigType_push((yyvsp[-7].type), (yyvsp[-6].decl).type);
		 /* Sneaky callback function trick */
		 if (SwigType_isfunction((yyvsp[-7].type))) {
		   SwigType_add_pointer((yyvsp[-7].type));
		 }
		 (yyval.node) = new_node("constant");
		 Setattr((yyval.node), "name", (yyvsp[-6].decl).id);
		 Setattr((yyval.node), "type", (yyvsp[-7].type));
		 Setattr((yyval.node), "value", (yyvsp[-1].dtype).val);
		 if ((yyvsp[-1].dtype).rawval) Setattr((yyval.node), "rawval", (yyvsp[-1].dtype).rawval);
		 Setattr((yyval.node), "storage", "%constant");
		 SetFlag((yyval.node), "feature:immutable");
		 add_symbols((yyval.node));
	       }
#line 5267 "CParse/parser.c"
    break;

  case 46: /* constant_directive: CONSTANT error SEMI  */
#line 2097 "./CParse/parser.y"
                                     {
		 Swig_warning(WARN_PARSE_BAD_VALUE,cparse_file,cparse_line,"Bad constant value (ignored).\n");
		 (yyval.node) = 0;
	       }
#line 5276 "CParse/parser.c"
    break;

  case 47: /* constant_directive: CONSTANT error END  */
#line 2101 "./CParse/parser.y"
                                    {
		 Swig_error(cparse_file,cparse_line,"Missing semicolon (';') after %%constant.\n");
		 Exit(EXIT_FAILURE);
	       }
#line 5285 "CParse/parser.c"
    break;

  case 48: /* echo_directive: ECHO HBLOCK  */
#line 2112 "./CParse/parser.y"
                             {
		 char temp[64];
		 Replace((yyvsp[0].str),"$file",cparse_file, DOH_REPLACE_ANY);
		 sprintf(temp,"%d", cparse_line);
		 Replace((yyvsp[0].str),"$line",temp,DOH_REPLACE_ANY);
		 Printf(stderr,"%s\n", (yyvsp[0].str));
		 Delete((yyvsp[0].str));
                 (yyval.node) = 0;
	       }
#line 5299 "CParse/parser.c"
    break;

  case 49: /* echo_directive: ECHO string  */
#line 2121 "./CParse/parser.y"
                             {
		 char temp[64];
		 String *s = (yyvsp[0].str);
		 Replace(s,"$file",cparse_file, DOH_REPLACE_ANY);
		 sprintf(temp,"%d", cparse_line);
		 Replace(s,"$line",temp,DOH_REPLACE_ANY);
		 Printf(stderr,"%s\n", s);
		 Delete(s);
                 (yyval.node) = 0;
               }
#line 5314 "CParse/parser.c"
    break;

  case 50: /* stringtype: string LBRACE parm RBRACE  */
#line 2134 "./CParse/parser.y"
                                          {		 
                 (yyval.node) = NewHash();
                 Setattr((yyval.node),"value",(yyvsp[-3].str));
		 Setattr((yyval.node),"type",Getattr((yyvsp[-1].p),"type"));
               }
#line 5324 "CParse/parser.c"
    break;

  case 51: /* fname: string  */
#line 2141 "./CParse/parser.y"
                       {
                 (yyval.node) = NewHash();
                 Setattr((yyval.node),"value",(yyvsp[0].str));
              }
#line 5333 "CParse/parser.c"
    break;

  case 53: /* fragment_directive: FRAGMENT LPAREN fname COMMA kwargs RPAREN HBLOCK  */
#line 2156 "./CParse/parser.y"
                                                                     {
                   Hash *p = (yyvsp[-2].node);
		   (yyval.node) = new_node("fragment");
		   Setattr((yyval.node),"value",Getattr((yyvsp[-4].node),"value"));
		   Setattr((yyval.node),"type",Getattr((yyvsp[-4].node),"type"));
		   Setattr((yyval.node),"section",Getattr(p,"name"));
		   Setattr((yyval.node),"kwargs",nextSibling(p));
		   Setattr((yyval.node),"code",(yyvsp[0].str));
                 }
#line 5347 "CParse/parser.c"
    break;

  case 54: /* fragment_directive: FRAGMENT LPAREN fname COMMA kwargs RPAREN LBRACE  */
#line 2165 "./CParse/parser.y"
                                                                    {
		   Hash *p = (yyvsp[-2].node);
		   String *code;
		   if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		   (yyval.node) = new_node("fragment");
		   Setattr((yyval.node),"value",Getattr((yyvsp[-4].node),"value"));
		   Setattr((yyval.node),"type",Getattr((yyvsp[-4].node),"type"));
		   Setattr((yyval.node),"section",Getattr(p,"name"));
		   Setattr((yyval.node),"kwargs",nextSibling(p));
		   Delitem(scanner_ccode,0);
		   Delitem(scanner_ccode,DOH_END);
		   code = Copy(scanner_ccode);
		   Setattr((yyval.node),"code",code);
		   Delete(code);
                 }
#line 5367 "CParse/parser.c"
    break;

  case 55: /* fragment_directive: FRAGMENT LPAREN fname RPAREN SEMI  */
#line 2180 "./CParse/parser.y"
                                                     {
		   (yyval.node) = new_node("fragment");
		   Setattr((yyval.node),"value",Getattr((yyvsp[-2].node),"value"));
		   Setattr((yyval.node),"type",Getattr((yyvsp[-2].node),"type"));
		   Setattr((yyval.node),"emitonly","1");
		 }
#line 5378 "CParse/parser.c"
    break;

  case 56: /* @2: %empty  */
#line 2193 "./CParse/parser.y"
                                                             {
		     (yyval.loc).filename = Copy(cparse_file);
		     (yyval.loc).line = cparse_line;
		     scanner_set_location((yyvsp[-1].str),1);
                     if ((yyvsp[-2].node)) { 
		       String *maininput = Getattr((yyvsp[-2].node), "maininput");
		       if (maininput)
		         scanner_set_main_input_file(NewString(maininput));
		     }
               }
#line 5393 "CParse/parser.c"
    break;

  case 57: /* include_directive: includetype options string BEGINFILE @2 interface ENDOFFILE  */
#line 2202 "./CParse/parser.y"
                                     {
                     String *mname = 0;
                     (yyval.node) = (yyvsp[-1].node);
		     scanner_set_location((yyvsp[-2].loc).filename, (yyvsp[-2].loc).line + 1);
		     if (strcmp((yyvsp[-6].loc).type,"include") == 0) set_nodeType((yyval.node),"include");
		     if (strcmp((yyvsp[-6].loc).type,"import") == 0) {
		       mname = (yyvsp[-5].node) ? Getattr((yyvsp[-5].node),"module") : 0;
		       set_nodeType((yyval.node),"import");
		       if (import_mode) --import_mode;
		     }
		     
		     Setattr((yyval.node),"name",(yyvsp[-4].str));
		     /* Search for the module (if any) */
		     {
			 Node *n = firstChild((yyval.node));
			 while (n) {
			     if (Strcmp(nodeType(n),"module") == 0) {
			         if (mname) {
				   Setattr(n,"name", mname);
				   mname = 0;
				 }
				 Setattr((yyval.node),"module",Getattr(n,"name"));
				 break;
			     }
			     n = nextSibling(n);
			 }
			 if (mname) {
			   /* There is no module node in the import
			      node, ie, you imported a .h file
			      directly.  We are forced then to create
			      a new import node with a module node.
			   */			      
			   Node *nint = new_node("import");
			   Node *mnode = new_node("module");
			   Setattr(mnode,"name", mname);
                           Setattr(mnode,"options",(yyvsp[-5].node));
			   appendChild(nint,mnode);
			   Delete(mnode);
			   appendChild(nint,firstChild((yyval.node)));
			   (yyval.node) = nint;
			   Setattr((yyval.node),"module",mname);
			 }
		     }
		     Setattr((yyval.node),"options",(yyvsp[-5].node));
               }
#line 5443 "CParse/parser.c"
    break;

  case 58: /* includetype: INCLUDE  */
#line 2249 "./CParse/parser.y"
                         { (yyval.loc).type = "include"; }
#line 5449 "CParse/parser.c"
    break;

  case 59: /* includetype: IMPORT  */
#line 2250 "./CParse/parser.y"
                         { (yyval.loc).type = "import"; ++import_mode;}
#line 5455 "CParse/parser.c"
    break;

  case 60: /* inline_directive: INLINE HBLOCK  */
#line 2257 "./CParse/parser.y"
                                 {
                 String *cpps;
		 if (Namespaceprefix) {
		   Swig_error(cparse_file, cparse_start_line, "%%inline directive inside a namespace is disallowed.\n");
		   (yyval.node) = 0;
		 } else {
		   (yyval.node) = new_node("insert");
		   Setattr((yyval.node),"code",(yyvsp[0].str));
		   /* Need to run through the preprocessor */
		   Seek((yyvsp[0].str),0,SEEK_SET);
		   Setline((yyvsp[0].str),cparse_start_line);
		   Setfile((yyvsp[0].str),cparse_file);
		   cpps = Preprocessor_parse((yyvsp[0].str));
		   start_inline(Char(cpps), cparse_start_line);
		   Delete((yyvsp[0].str));
		   Delete(cpps);
		 }
		 
	       }
#line 5479 "CParse/parser.c"
    break;

  case 61: /* inline_directive: INLINE LBRACE  */
#line 2276 "./CParse/parser.y"
                               {
                 String *cpps;
		 int start_line = cparse_line;
		 if (Namespaceprefix) {
		   Swig_error(cparse_file, cparse_start_line, "%%inline directive inside a namespace is disallowed.\n");
		 }
		 if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		 if (Namespaceprefix) {
		   (yyval.node) = 0;
		 } else {
		   String *code;
                   (yyval.node) = new_node("insert");
		   Delitem(scanner_ccode,0);
		   Delitem(scanner_ccode,DOH_END);
		   code = Copy(scanner_ccode);
		   Setattr((yyval.node),"code", code);
		   Delete(code);		   
		   cpps=Copy(scanner_ccode);
		   start_inline(Char(cpps), start_line);
		   Delete(cpps);
		 }
               }
#line 5506 "CParse/parser.c"
    break;

  case 62: /* insert_directive: HBLOCK  */
#line 2308 "./CParse/parser.y"
                          {
                 (yyval.node) = new_node("insert");
		 Setattr((yyval.node),"code",(yyvsp[0].str));
	       }
#line 5515 "CParse/parser.c"
    break;

  case 63: /* insert_directive: INSERT LPAREN idstring RPAREN string  */
#line 2312 "./CParse/parser.y"
                                                      {
		 String *code = NewStringEmpty();
		 (yyval.node) = new_node("insert");
		 Setattr((yyval.node),"section",(yyvsp[-2].id));
		 Setattr((yyval.node),"code",code);
		 if (Swig_insert_file((yyvsp[0].str),code) < 0) {
		   Swig_error(cparse_file, cparse_line, "Couldn't find '%s'.\n", (yyvsp[0].str));
		   (yyval.node) = 0;
		 } 
               }
#line 5530 "CParse/parser.c"
    break;

  case 64: /* insert_directive: INSERT LPAREN idstring RPAREN HBLOCK  */
#line 2322 "./CParse/parser.y"
                                                      {
		 (yyval.node) = new_node("insert");
		 Setattr((yyval.node),"section",(yyvsp[-2].id));
		 Setattr((yyval.node),"code",(yyvsp[0].str));
               }
#line 5540 "CParse/parser.c"
    break;

  case 65: /* insert_directive: INSERT LPAREN idstring RPAREN LBRACE  */
#line 2327 "./CParse/parser.y"
                                                      {
		 String *code;
		 if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		 (yyval.node) = new_node("insert");
		 Setattr((yyval.node),"section",(yyvsp[-2].id));
		 Delitem(scanner_ccode,0);
		 Delitem(scanner_ccode,DOH_END);
		 code = Copy(scanner_ccode);
		 Setattr((yyval.node),"code", code);
		 Delete(code);
	       }
#line 5556 "CParse/parser.c"
    break;

  case 66: /* module_directive: MODULE options idstring  */
#line 2345 "./CParse/parser.y"
                                          {
                 (yyval.node) = new_node("module");
		 if ((yyvsp[-1].node)) {
		   Setattr((yyval.node),"options",(yyvsp[-1].node));
		   if (Getattr((yyvsp[-1].node),"directors")) {
		     Wrapper_director_mode_set(1);
		     if (!cparse_cplusplus) {
		       Swig_error(cparse_file, cparse_line, "Directors are not supported for C code and require the -c++ option\n");
		     }
		   } 
		   if (Getattr((yyvsp[-1].node),"dirprot")) {
		     Wrapper_director_protected_mode_set(1);
		   } 
		   if (Getattr((yyvsp[-1].node),"allprotected")) {
		     Wrapper_all_protected_mode_set(1);
		   } 
		   if (Getattr((yyvsp[-1].node),"templatereduce")) {
		     template_reduce = 1;
		   }
		   if (Getattr((yyvsp[-1].node),"notemplatereduce")) {
		     template_reduce = 0;
		   }
		 }
		 if (!ModuleName) ModuleName = NewString((yyvsp[0].id));
		 if (!import_mode) {
		   /* first module included, we apply global
		      ModuleName, which can be modify by -module */
		   String *mname = Copy(ModuleName);
		   Setattr((yyval.node),"name",mname);
		   Delete(mname);
		 } else { 
		   /* import mode, we just pass the idstring */
		   Setattr((yyval.node),"name",(yyvsp[0].id));   
		 }		 
		 if (!module_node) module_node = (yyval.node);
	       }
#line 5597 "CParse/parser.c"
    break;

  case 67: /* native_directive: NATIVE LPAREN identifier RPAREN storage_class identifier SEMI  */
#line 2388 "./CParse/parser.y"
                                                                                 {
                 (yyval.node) = new_node("native");
		 Setattr((yyval.node),"name",(yyvsp[-4].id));
		 Setattr((yyval.node),"wrap:name",(yyvsp[-1].id));
		 Delete((yyvsp[-2].str));
	         add_symbols((yyval.node));
	       }
#line 5609 "CParse/parser.c"
    break;

  case 68: /* native_directive: NATIVE LPAREN identifier RPAREN storage_class type declarator SEMI  */
#line 2395 "./CParse/parser.y"
                                                                                    {
		 if (!SwigType_isfunction((yyvsp[-1].decl).type)) {
		   Swig_error(cparse_file,cparse_line,"%%native declaration '%s' is not a function.\n", (yyvsp[-1].decl).id);
		   (yyval.node) = 0;
		 } else {
		     Delete(SwigType_pop_function((yyvsp[-1].decl).type));
		     /* Need check for function here */
		     SwigType_push((yyvsp[-2].type),(yyvsp[-1].decl).type);
		     (yyval.node) = new_node("native");
	             Setattr((yyval.node),"name",(yyvsp[-5].id));
		     Setattr((yyval.node),"wrap:name",(yyvsp[-1].decl).id);
		     Setattr((yyval.node),"type",(yyvsp[-2].type));
		     Setattr((yyval.node),"parms",(yyvsp[-1].decl).parms);
		     Setattr((yyval.node),"decl",(yyvsp[-1].decl).type);
		 }
		 Delete((yyvsp[-3].str));
	         add_symbols((yyval.node));
	       }
#line 5632 "CParse/parser.c"
    break;

  case 69: /* pragma_directive: PRAGMA pragma_lang identifier EQUAL pragma_arg  */
#line 2422 "./CParse/parser.y"
                                                                  {
                 (yyval.node) = new_node("pragma");
		 Setattr((yyval.node),"lang",(yyvsp[-3].id));
		 Setattr((yyval.node),"name",(yyvsp[-2].id));
		 Setattr((yyval.node),"value",(yyvsp[0].str));
	       }
#line 5643 "CParse/parser.c"
    break;

  case 70: /* pragma_directive: PRAGMA pragma_lang identifier  */
#line 2428 "./CParse/parser.y"
                                              {
		(yyval.node) = new_node("pragma");
		Setattr((yyval.node),"lang",(yyvsp[-1].id));
		Setattr((yyval.node),"name",(yyvsp[0].id));
	      }
#line 5653 "CParse/parser.c"
    break;

  case 73: /* pragma_lang: LPAREN identifier RPAREN  */
#line 2439 "./CParse/parser.y"
                                         { (yyval.id) = (yyvsp[-1].id); }
#line 5659 "CParse/parser.c"
    break;

  case 74: /* pragma_lang: %empty  */
#line 2440 "./CParse/parser.y"
                       { (yyval.id) = "swig"; }
#line 5665 "CParse/parser.c"
    break;

  case 75: /* rename_directive: rename_namewarn declarator idstring SEMI  */
#line 2447 "./CParse/parser.y"
                                                            {
                SwigType *t = (yyvsp[-2].decl).type;
		Hash *kws = NewHash();
		String *fixname;
		fixname = feature_identifier_fix((yyvsp[-2].decl).id);
		Setattr(kws,"name",(yyvsp[-1].id));
		if (!Len(t)) t = 0;
		/* Special declarator check */
		if (t) {
		  if (SwigType_isfunction(t)) {
		    SwigType *decl = SwigType_pop_function(t);
		    if (SwigType_ispointer(t)) {
		      String *nname = NewStringf("*%s",fixname);
		      if ((yyvsp[-3].intvalue)) {
			Swig_name_rename_add(Namespaceprefix, nname,decl,kws,(yyvsp[-2].decl).parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,nname,decl,kws);
		      }
		      Delete(nname);
		    } else {
		      if ((yyvsp[-3].intvalue)) {
			Swig_name_rename_add(Namespaceprefix,(fixname),decl,kws,(yyvsp[-2].decl).parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,(fixname),decl,kws);
		      }
		    }
		    Delete(decl);
		  } else if (SwigType_ispointer(t)) {
		    String *nname = NewStringf("*%s",fixname);
		    if ((yyvsp[-3].intvalue)) {
		      Swig_name_rename_add(Namespaceprefix,(nname),0,kws,(yyvsp[-2].decl).parms);
		    } else {
		      Swig_name_namewarn_add(Namespaceprefix,(nname),0,kws);
		    }
		    Delete(nname);
		  }
		} else {
		  if ((yyvsp[-3].intvalue)) {
		    Swig_name_rename_add(Namespaceprefix,(fixname),0,kws,(yyvsp[-2].decl).parms);
		  } else {
		    Swig_name_namewarn_add(Namespaceprefix,(fixname),0,kws);
		  }
		}
                (yyval.node) = 0;
		scanner_clear_rename();
              }
#line 5716 "CParse/parser.c"
    break;

  case 76: /* rename_directive: rename_namewarn LPAREN kwargs RPAREN declarator cpp_const SEMI  */
#line 2493 "./CParse/parser.y"
                                                                               {
		String *fixname;
		Hash *kws = (yyvsp[-4].node);
		SwigType *t = (yyvsp[-2].decl).type;
		fixname = feature_identifier_fix((yyvsp[-2].decl).id);
		if (!Len(t)) t = 0;
		/* Special declarator check */
		if (t) {
		  if ((yyvsp[-1].dtype).qualifier) SwigType_push(t,(yyvsp[-1].dtype).qualifier);
		  if (SwigType_isfunction(t)) {
		    SwigType *decl = SwigType_pop_function(t);
		    if (SwigType_ispointer(t)) {
		      String *nname = NewStringf("*%s",fixname);
		      if ((yyvsp[-6].intvalue)) {
			Swig_name_rename_add(Namespaceprefix, nname,decl,kws,(yyvsp[-2].decl).parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,nname,decl,kws);
		      }
		      Delete(nname);
		    } else {
		      if ((yyvsp[-6].intvalue)) {
			Swig_name_rename_add(Namespaceprefix,(fixname),decl,kws,(yyvsp[-2].decl).parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,(fixname),decl,kws);
		      }
		    }
		    Delete(decl);
		  } else if (SwigType_ispointer(t)) {
		    String *nname = NewStringf("*%s",fixname);
		    if ((yyvsp[-6].intvalue)) {
		      Swig_name_rename_add(Namespaceprefix,(nname),0,kws,(yyvsp[-2].decl).parms);
		    } else {
		      Swig_name_namewarn_add(Namespaceprefix,(nname),0,kws);
		    }
		    Delete(nname);
		  }
		} else {
		  if ((yyvsp[-6].intvalue)) {
		    Swig_name_rename_add(Namespaceprefix,(fixname),0,kws,(yyvsp[-2].decl).parms);
		  } else {
		    Swig_name_namewarn_add(Namespaceprefix,(fixname),0,kws);
		  }
		}
                (yyval.node) = 0;
		scanner_clear_rename();
              }
#line 5767 "CParse/parser.c"
    break;

  case 77: /* rename_directive: rename_namewarn LPAREN kwargs RPAREN string SEMI  */
#line 2539 "./CParse/parser.y"
                                                                 {
		if ((yyvsp[-5].intvalue)) {
		  Swig_name_rename_add(Namespaceprefix,(yyvsp[-1].str),0,(yyvsp[-3].node),0);
		} else {
		  Swig_name_namewarn_add(Namespaceprefix,(yyvsp[-1].str),0,(yyvsp[-3].node));
		}
		(yyval.node) = 0;
		scanner_clear_rename();
              }
#line 5781 "CParse/parser.c"
    break;

  case 78: /* rename_namewarn: RENAME  */
#line 2550 "./CParse/parser.y"
                         {
		    (yyval.intvalue) = 1;
                }
#line 5789 "CParse/parser.c"
    break;

  case 79: /* rename_namewarn: NAMEWARN  */
#line 2553 "./CParse/parser.y"
                           {
                    (yyval.intvalue) = 0;
                }
#line 5797 "CParse/parser.c"
    break;

  case 80: /* feature_directive: FEATURE LPAREN idstring RPAREN declarator cpp_const stringbracesemi  */
#line 2580 "./CParse/parser.y"
                                                                                        {
                    String *val = (yyvsp[0].str) ? NewString((yyvsp[0].str)) : NewString("1");
                    new_feature((yyvsp[-4].id), val, 0, (yyvsp[-2].decl).id, (yyvsp[-2].decl).type, (yyvsp[-2].decl).parms, (yyvsp[-1].dtype).qualifier);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5808 "CParse/parser.c"
    break;

  case 81: /* feature_directive: FEATURE LPAREN idstring COMMA stringnum RPAREN declarator cpp_const SEMI  */
#line 2586 "./CParse/parser.y"
                                                                                             {
                    String *val = Len((yyvsp[-4].str)) ? (yyvsp[-4].str) : 0;
                    new_feature((yyvsp[-6].id), val, 0, (yyvsp[-2].decl).id, (yyvsp[-2].decl).type, (yyvsp[-2].decl).parms, (yyvsp[-1].dtype).qualifier);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5819 "CParse/parser.c"
    break;

  case 82: /* feature_directive: FEATURE LPAREN idstring featattr RPAREN declarator cpp_const stringbracesemi  */
#line 2592 "./CParse/parser.y"
                                                                                                 {
                    String *val = (yyvsp[0].str) ? NewString((yyvsp[0].str)) : NewString("1");
                    new_feature((yyvsp[-5].id), val, (yyvsp[-4].node), (yyvsp[-2].decl).id, (yyvsp[-2].decl).type, (yyvsp[-2].decl).parms, (yyvsp[-1].dtype).qualifier);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5830 "CParse/parser.c"
    break;

  case 83: /* feature_directive: FEATURE LPAREN idstring COMMA stringnum featattr RPAREN declarator cpp_const SEMI  */
#line 2598 "./CParse/parser.y"
                                                                                                      {
                    String *val = Len((yyvsp[-5].str)) ? (yyvsp[-5].str) : 0;
                    new_feature((yyvsp[-7].id), val, (yyvsp[-4].node), (yyvsp[-2].decl).id, (yyvsp[-2].decl).type, (yyvsp[-2].decl).parms, (yyvsp[-1].dtype).qualifier);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5841 "CParse/parser.c"
    break;

  case 84: /* feature_directive: FEATURE LPAREN idstring RPAREN stringbracesemi  */
#line 2606 "./CParse/parser.y"
                                                                   {
                    String *val = (yyvsp[0].str) ? NewString((yyvsp[0].str)) : NewString("1");
                    new_feature((yyvsp[-2].id), val, 0, 0, 0, 0, 0);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5852 "CParse/parser.c"
    break;

  case 85: /* feature_directive: FEATURE LPAREN idstring COMMA stringnum RPAREN SEMI  */
#line 2612 "./CParse/parser.y"
                                                                        {
                    String *val = Len((yyvsp[-2].str)) ? (yyvsp[-2].str) : 0;
                    new_feature((yyvsp[-4].id), val, 0, 0, 0, 0, 0);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5863 "CParse/parser.c"
    break;

  case 86: /* feature_directive: FEATURE LPAREN idstring featattr RPAREN stringbracesemi  */
#line 2618 "./CParse/parser.y"
                                                                            {
                    String *val = (yyvsp[0].str) ? NewString((yyvsp[0].str)) : NewString("1");
                    new_feature((yyvsp[-3].id), val, (yyvsp[-2].node), 0, 0, 0, 0);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5874 "CParse/parser.c"
    break;

  case 87: /* feature_directive: FEATURE LPAREN idstring COMMA stringnum featattr RPAREN SEMI  */
#line 2624 "./CParse/parser.y"
                                                                                 {
                    String *val = Len((yyvsp[-3].str)) ? (yyvsp[-3].str) : 0;
                    new_feature((yyvsp[-5].id), val, (yyvsp[-2].node), 0, 0, 0, 0);
                    (yyval.node) = 0;
                    scanner_clear_rename();
                  }
#line 5885 "CParse/parser.c"
    break;

  case 89: /* stringbracesemi: SEMI  */
#line 2633 "./CParse/parser.y"
                       { (yyval.str) = 0; }
#line 5891 "CParse/parser.c"
    break;

  case 90: /* stringbracesemi: PARMS LPAREN parms RPAREN SEMI  */
#line 2634 "./CParse/parser.y"
                                                 { (yyval.str) = (yyvsp[-2].pl); }
#line 5897 "CParse/parser.c"
    break;

  case 91: /* featattr: COMMA idstring EQUAL stringnum  */
#line 2637 "./CParse/parser.y"
                                                 {
		  (yyval.node) = NewHash();
		  Setattr((yyval.node),"name",(yyvsp[-2].id));
		  Setattr((yyval.node),"value",(yyvsp[0].str));
                }
#line 5907 "CParse/parser.c"
    break;

  case 92: /* featattr: COMMA idstring EQUAL stringnum featattr  */
#line 2642 "./CParse/parser.y"
                                                          {
		  (yyval.node) = NewHash();
		  Setattr((yyval.node),"name",(yyvsp[-3].id));
		  Setattr((yyval.node),"value",(yyvsp[-1].str));
                  set_nextSibling((yyval.node),(yyvsp[0].node));
                }
#line 5918 "CParse/parser.c"
    break;

  case 93: /* varargs_directive: VARARGS LPAREN varargs_parms RPAREN declarator cpp_const SEMI  */
#line 2652 "./CParse/parser.y"
                                                                                  {
                 Parm *val;
		 String *name;
		 SwigType *t;
		 if (Namespaceprefix) name = NewStringf("%s::%s", Namespaceprefix, (yyvsp[-2].decl).id);
		 else name = NewString((yyvsp[-2].decl).id);
		 val = (yyvsp[-4].pl);
		 if ((yyvsp[-2].decl).parms) {
		   Setmeta(val,"parms",(yyvsp[-2].decl).parms);
		 }
		 t = (yyvsp[-2].decl).type;
		 if (!Len(t)) t = 0;
		 if (t) {
		   if ((yyvsp[-1].dtype).qualifier) SwigType_push(t,(yyvsp[-1].dtype).qualifier);
		   if (SwigType_isfunction(t)) {
		     SwigType *decl = SwigType_pop_function(t);
		     if (SwigType_ispointer(t)) {
		       String *nname = NewStringf("*%s",name);
		       Swig_feature_set(Swig_cparse_features(), nname, decl, "feature:varargs", val, 0);
		       Delete(nname);
		     } else {
		       Swig_feature_set(Swig_cparse_features(), name, decl, "feature:varargs", val, 0);
		     }
		     Delete(decl);
		   } else if (SwigType_ispointer(t)) {
		     String *nname = NewStringf("*%s",name);
		     Swig_feature_set(Swig_cparse_features(),nname,0,"feature:varargs",val, 0);
		     Delete(nname);
		   }
		 } else {
		   Swig_feature_set(Swig_cparse_features(),name,0,"feature:varargs",val, 0);
		 }
		 Delete(name);
		 (yyval.node) = 0;
              }
#line 5958 "CParse/parser.c"
    break;

  case 95: /* varargs_parms: NUM_INT COMMA parm  */
#line 2689 "./CParse/parser.y"
                                     { 
		  int i;
		  int n;
		  Parm *p;
		  n = atoi(Char((yyvsp[-2].dtype).val));
		  if (n <= 0) {
		    Swig_error(cparse_file, cparse_line,"Argument count in %%varargs must be positive.\n");
		    (yyval.pl) = 0;
		  } else {
		    String *name = Getattr((yyvsp[0].p), "name");
		    (yyval.pl) = Copy((yyvsp[0].p));
		    if (name)
		      Setattr((yyval.pl), "name", NewStringf("%s%d", name, n));
		    for (i = 1; i < n; i++) {
		      p = Copy((yyvsp[0].p));
		      name = Getattr(p, "name");
		      if (name)
		        Setattr(p, "name", NewStringf("%s%d", name, n-i));
		      set_nextSibling(p,(yyval.pl));
		      Delete((yyval.pl));
		      (yyval.pl) = p;
		    }
		  }
                }
#line 5987 "CParse/parser.c"
    break;

  case 96: /* typemap_directive: TYPEMAP LPAREN typemap_type RPAREN tm_list stringbrace  */
#line 2724 "./CParse/parser.y"
                                                                            {
		   (yyval.node) = 0;
		   if ((yyvsp[-3].tmap).method) {
		     String *code = 0;
		     (yyval.node) = new_node("typemap");
		     Setattr((yyval.node),"method",(yyvsp[-3].tmap).method);
		     if ((yyvsp[-3].tmap).kwargs) {
		       ParmList *kw = (yyvsp[-3].tmap).kwargs;
                       code = remove_block(kw, (yyvsp[0].str));
		       Setattr((yyval.node),"kwargs", (yyvsp[-3].tmap).kwargs);
		     }
		     code = code ? code : NewString((yyvsp[0].str));
		     Setattr((yyval.node),"code", code);
		     Delete(code);
		     appendChild((yyval.node),(yyvsp[-1].p));
		   }
	       }
#line 6009 "CParse/parser.c"
    break;

  case 97: /* typemap_directive: TYPEMAP LPAREN typemap_type RPAREN tm_list SEMI  */
#line 2741 "./CParse/parser.y"
                                                                 {
		 (yyval.node) = 0;
		 if ((yyvsp[-3].tmap).method) {
		   (yyval.node) = new_node("typemap");
		   Setattr((yyval.node),"method",(yyvsp[-3].tmap).method);
		   appendChild((yyval.node),(yyvsp[-1].p));
		 }
	       }
#line 6022 "CParse/parser.c"
    break;

  case 98: /* typemap_directive: TYPEMAP LPAREN typemap_type RPAREN tm_list EQUAL typemap_parm SEMI  */
#line 2749 "./CParse/parser.y"
                                                                                    {
		   (yyval.node) = 0;
		   if ((yyvsp[-5].tmap).method) {
		     (yyval.node) = new_node("typemapcopy");
		     Setattr((yyval.node),"method",(yyvsp[-5].tmap).method);
		     Setattr((yyval.node),"pattern", Getattr((yyvsp[-1].p),"pattern"));
		     appendChild((yyval.node),(yyvsp[-3].p));
		   }
	       }
#line 6036 "CParse/parser.c"
    break;

  case 99: /* typemap_type: kwargs  */
#line 2762 "./CParse/parser.y"
                        {
		 String *name = Getattr((yyvsp[0].node), "name");
		 Hash *p = nextSibling((yyvsp[0].node));
		 (yyval.tmap).method = name;
		 (yyval.tmap).kwargs = p;
		 if (Getattr((yyvsp[0].node), "value")) {
		   Swig_error(cparse_file, cparse_line,
			      "%%typemap method shouldn't have a value specified.\n");
		 }
		 while (p) {
		   if (!Getattr(p, "value")) {
		     Swig_error(cparse_file, cparse_line,
				"%%typemap attribute '%s' is missing its value.  If this is specifying the target language, that's no longer supported: use #ifdef SWIG<LANG> instead.\n",
				Getattr(p, "name"));
		     /* Set to empty value to avoid segfaults later. */
		     Setattr(p, "value", NewStringEmpty());
		   }
		   p = nextSibling(p);
		 }
                }
#line 6061 "CParse/parser.c"
    break;

  case 100: /* tm_list: typemap_parm tm_tail  */
#line 2784 "./CParse/parser.y"
                                      {
                 (yyval.p) = (yyvsp[-1].p);
		 set_nextSibling((yyval.p),(yyvsp[0].p));
		}
#line 6070 "CParse/parser.c"
    break;

  case 101: /* tm_tail: COMMA typemap_parm tm_tail  */
#line 2790 "./CParse/parser.y"
                                            {
                 (yyval.p) = (yyvsp[-1].p);
		 set_nextSibling((yyval.p),(yyvsp[0].p));
                }
#line 6079 "CParse/parser.c"
    break;

  case 102: /* tm_tail: %empty  */
#line 2794 "./CParse/parser.y"
                        { (yyval.p) = 0;}
#line 6085 "CParse/parser.c"
    break;

  case 103: /* typemap_parm: type plain_declarator  */
#line 2797 "./CParse/parser.y"
                                       {
                  Parm *parm;
		  SwigType_push((yyvsp[-1].type),(yyvsp[0].decl).type);
		  (yyval.p) = new_node("typemapitem");
		  parm = NewParmWithoutFileLineInfo((yyvsp[-1].type),(yyvsp[0].decl).id);
		  Setattr((yyval.p),"pattern",parm);
		  Setattr((yyval.p),"parms", (yyvsp[0].decl).parms);
		  Delete(parm);
		  /*		  $$ = NewParmWithoutFileLineInfo($1,$2.id);
				  Setattr($$,"parms",$2.parms); */
                }
#line 6101 "CParse/parser.c"
    break;

  case 104: /* typemap_parm: LPAREN parms RPAREN  */
#line 2808 "./CParse/parser.y"
                                     {
                  (yyval.p) = new_node("typemapitem");
		  Setattr((yyval.p),"pattern",(yyvsp[-1].pl));
		  /*		  Setattr($$,"multitype",$2); */
               }
#line 6111 "CParse/parser.c"
    break;

  case 105: /* typemap_parm: LPAREN parms RPAREN LPAREN parms RPAREN  */
#line 2813 "./CParse/parser.y"
                                                         {
		 (yyval.p) = new_node("typemapitem");
		 Setattr((yyval.p),"pattern", (yyvsp[-4].pl));
		 /*                 Setattr($$,"multitype",$2); */
		 Setattr((yyval.p),"parms",(yyvsp[-1].pl));
               }
#line 6122 "CParse/parser.c"
    break;

  case 106: /* types_directive: TYPES LPAREN parms RPAREN stringbracesemi  */
#line 2826 "./CParse/parser.y"
                                                            {
                   (yyval.node) = new_node("types");
		   Setattr((yyval.node),"parms",(yyvsp[-2].pl));
                   if ((yyvsp[0].str))
		     Setattr((yyval.node),"convcode",NewString((yyvsp[0].str)));
               }
#line 6133 "CParse/parser.c"
    break;

  case 107: /* template_directive: SWIGTEMPLATE LPAREN idstringopt RPAREN idcolonnt LESSTHAN valparms GREATERTHAN SEMI  */
#line 2838 "./CParse/parser.y"
                                                                                                        {
                  Parm *p;
		  Node *n = 0;
		  Node *outer_class = currentOuterClass;
		  Symtab *tscope = 0;
		  String *symname = (yyvsp[-6].id) ? NewString((yyvsp[-6].id)) : 0;
		  int errored_flag = 0;
		  String *idcolonnt;

		  (yyval.node) = 0;

		  tscope = Swig_symbol_current();          /* Get the current scope */

		  /* If the class name is qualified, we need to create or lookup namespace entries */
		  idcolonnt = resolve_create_node_scope((yyvsp[-4].str), 0, &errored_flag);

		  if (!errored_flag) {
		    if (nscope_inner && Strcmp(nodeType(nscope_inner), "class") == 0)
		      outer_class = nscope_inner;

		    /*
		      We use the new namespace entry 'nscope' only to
		      emit the template node. The template parameters are
		      resolved in the current 'tscope'.

		      This is closer to the C++ (typedef) behavior.
		    */
		    n = Swig_cparse_template_locate(idcolonnt, (yyvsp[-2].p), symname, tscope);
		  }

		  /* Patch the argument types to respect namespaces */
		  p = (yyvsp[-2].p);
		  while (p) {
		    SwigType *value = Getattr(p,"value");
		    if (!value) {
		      SwigType *ty = Getattr(p,"type");
		      if (ty) {
			SwigType *rty = 0;
			int reduce = template_reduce;
			if (reduce || !SwigType_ispointer(ty)) {
			  rty = Swig_symbol_typedef_reduce(ty,tscope);
			  if (!reduce) reduce = SwigType_ispointer(rty);
			}
			ty = reduce ? Swig_symbol_type_qualify(rty,tscope) : Swig_symbol_type_qualify(ty,tscope);
			Setattr(p,"type",ty);
			Delete(ty);
			Delete(rty);
		      }
		    } else {
		      value = Swig_symbol_type_qualify(value,tscope);
		      Setattr(p,"value",value);
		      Delete(value);
		    }

		    p = nextSibling(p);
		  }

		  /* Look for the template */
		  {
                    Node *nn = n;
                    Node *linklistend = 0;
                    Node *linkliststart = 0;
                    while (nn) {
                      Node *templnode = 0;
                      if (GetFlag(nn, "instantiate")) {
			Delattr(nn, "instantiate");
			{
			  int nnisclass = (Strcmp(Getattr(nn, "templatetype"), "class") == 0); /* if not a class template it is a function template */
			  Parm *tparms = Getattr(nn, "templateparms");
			  int specialized = !tparms; /* fully specialized (an explicit specialization) */
			  String *tname = Copy(idcolonnt);
			  Node *primary_template = Swig_symbol_clookup(tname, 0);

			  /* Expand the template */
			  ParmList *temparms = Swig_cparse_template_parms_expand((yyvsp[-2].p), primary_template, nn);

                          templnode = copy_node(nn);
			  update_nested_classes(templnode); /* update classes nested within template */
                          /* We need to set the node name based on name used to instantiate */
                          Setattr(templnode,"name",tname);
			  Delete(tname);
                          if (!specialized) {
                            Delattr(templnode,"sym:typename");
                          } else {
                            Setattr(templnode,"sym:typename","1");
                          }
			  /* for now, nested %template is allowed only in the same scope as the template declaration */
                          if (symname && !(nnisclass && ((outer_class && (outer_class != Getattr(nn, "nested:outer")))
			    ||(extendmode && current_class && (current_class != Getattr(nn, "nested:outer")))))) {
			    /*
			       Comment this out for 1.3.28. We need to
			       re-enable it later but first we need to
			       move %ignore from using %rename to use
			       %feature(ignore).

			       String *symname = Swig_name_make(templnode, 0, symname, 0, 0);
			    */
                            Swig_cparse_template_expand(templnode, symname, temparms, tscope);
                            Setattr(templnode, "sym:name", symname);
                          } else {
                            static int cnt = 0;
                            String *nname = NewStringf("__dummy_%d__", cnt++);
                            Swig_cparse_template_expand(templnode,nname,temparms,tscope);
                            Setattr(templnode,"sym:name",nname);
                            SetFlag(templnode,"hidden");
			    Delete(nname);
                            Setattr(templnode,"feature:onlychildren", "typemap,typemapitem,typemapcopy,typedef,types,fragment,apply");
			    if (symname) {
			      Swig_warning(WARN_PARSE_NESTED_TEMPLATE, cparse_file, cparse_line, "Named nested template instantiations not supported. Processing as if no name was given to %%template().\n");
			    }
                          }
                          Delattr(templnode,"templatetype");
                          Setattr(templnode,"template",nn);
                          Setfile(templnode,cparse_file);
                          Setline(templnode,cparse_line);
                          Delete(temparms);
			  if (outer_class && nnisclass) {
			    SetFlag(templnode, "nested");
			    Setattr(templnode, "nested:outer", outer_class);
			  }
                          add_symbols_copy(templnode);

                          if (Strcmp(nodeType(templnode),"class") == 0) {

                            /* Identify pure abstract methods */
                            Setattr(templnode,"abstracts", pure_abstracts(firstChild(templnode)));

                            /* Set up inheritance in symbol table */
                            {
                              Symtab  *csyms;
                              List *baselist = Getattr(templnode,"baselist");
                              csyms = Swig_symbol_current();
                              Swig_symbol_setscope(Getattr(templnode,"symtab"));
                              if (baselist) {
                                List *bases = Swig_make_inherit_list(Getattr(templnode,"name"),baselist, Namespaceprefix);
                                if (bases) {
                                  Iterator s;
                                  for (s = First(bases); s.item; s = Next(s)) {
                                    Symtab *st = Getattr(s.item,"symtab");
                                    if (st) {
				      Setfile(st,Getfile(s.item));
				      Setline(st,Getline(s.item));
                                      Swig_symbol_inherit(st);
                                    }
                                  }
				  Delete(bases);
                                }
                              }
                              Swig_symbol_setscope(csyms);
                            }

                            /* Merge in %extend methods for this class.
			       This only merges methods within %extend for a template specialized class such as
			         template<typename T> class K {}; %extend K<int> { ... }
			       The copy_node() call above has already added in the generic %extend methods such as
			         template<typename T> class K {}; %extend K { ... } */

			    /* !!! This may be broken.  We may have to add the
			       %extend methods at the beginning of the class */
                            {
                              String *stmp = 0;
                              String *clsname;
                              Node *am;
                              if (Namespaceprefix) {
                                clsname = stmp = NewStringf("%s::%s", Namespaceprefix, Getattr(templnode,"name"));
                              } else {
                                clsname = Getattr(templnode,"name");
                              }
                              am = Getattr(Swig_extend_hash(),clsname);
                              if (am) {
                                Symtab *st = Swig_symbol_current();
                                Swig_symbol_setscope(Getattr(templnode,"symtab"));
                                /*			    Printf(stdout,"%s: %s %p %p\n", Getattr(templnode,"name"), clsname, Swig_symbol_current(), Getattr(templnode,"symtab")); */
                                Swig_extend_merge(templnode,am);
                                Swig_symbol_setscope(st);
				Swig_extend_append_previous(templnode,am);
                                Delattr(Swig_extend_hash(),clsname);
                              }
			      if (stmp) Delete(stmp);
                            }

                            /* Add to classes hash */
			    if (!classes)
			      classes = NewHash();

			    if (Namespaceprefix) {
			      String *temp = NewStringf("%s::%s", Namespaceprefix, Getattr(templnode,"name"));
			      Setattr(classes,temp,templnode);
			      Delete(temp);
			    } else {
			      String *qs = Swig_symbol_qualifiedscopename(templnode);
			      Setattr(classes, qs,templnode);
			      Delete(qs);
			    }
                          }
                        }

                        /* all the overloaded function templates are added into a linked list */
                        if (!linkliststart)
                          linkliststart = templnode;
                        if (nscope_inner) {
                          /* non-global namespace */
                          if (templnode) {
                            appendChild(nscope_inner,templnode);
			    Delete(templnode);
                            if (nscope) (yyval.node) = nscope;
                          }
                        } else {
                          /* global namespace */
                          if (!linklistend) {
                            (yyval.node) = templnode;
                          } else {
                            set_nextSibling(linklistend,templnode);
			    Delete(templnode);
                          }
                          linklistend = templnode;
                        }
                      }
                      nn = Getattr(nn,"sym:nextSibling"); /* repeat for overloaded function templates. If a class template there will never be a sibling. */
                    }
                    update_defaultargs(linkliststart);
                    update_abstracts(linkliststart);
		  }
	          Swig_symbol_setscope(tscope);
		  Delete(Namespaceprefix);
		  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		  Delete(symname);
                }
#line 6366 "CParse/parser.c"
    break;

  case 108: /* warn_directive: WARN string  */
#line 3073 "./CParse/parser.y"
                             {
		  Swig_warning(0,cparse_file, cparse_line,"%s\n", (yyvsp[0].str));
		  (yyval.node) = 0;
               }
#line 6375 "CParse/parser.c"
    break;

  case 109: /* c_declaration: c_decl  */
#line 3083 "./CParse/parser.y"
                         {
                    (yyval.node) = (yyvsp[0].node); 
                    if ((yyval.node)) {
   		      add_symbols((yyval.node));
                      default_arguments((yyval.node));
   	            }
                }
#line 6387 "CParse/parser.c"
    break;

  case 112: /* $@3: %empty  */
#line 3095 "./CParse/parser.y"
                                       {
		  if (Strcmp((yyvsp[-1].str),"C") == 0) {
		    cparse_externc = 1;
		  }
		}
#line 6397 "CParse/parser.c"
    break;

  case 113: /* c_declaration: EXTERN string LBRACE $@3 interface RBRACE  */
#line 3099 "./CParse/parser.y"
                                   {
		  cparse_externc = 0;
		  if (Strcmp((yyvsp[-4].str),"C") == 0) {
		    Node *n = firstChild((yyvsp[-1].node));
		    (yyval.node) = new_node("extern");
		    Setattr((yyval.node),"name",(yyvsp[-4].str));
		    appendChild((yyval.node),n);
		    while (n) {
		      String *s = Getattr(n, "storage");
		      if (s) {
			if (Strstr(s, "thread_local")) {
			  Insert(s,0,"externc ");
			} else if (!Equal(s, "typedef")) {
			  Setattr(n,"storage","externc");
			}
		      } else {
			Setattr(n,"storage","externc");
		      }
		      n = nextSibling(n);
		    }
		  } else {
		    if (!Equal((yyvsp[-4].str),"C++")) {
		      Swig_warning(WARN_PARSE_UNDEFINED_EXTERN,cparse_file, cparse_line,"Unrecognized extern type \"%s\".\n", (yyvsp[-4].str));
		    }
		    (yyval.node) = new_node("extern");
		    Setattr((yyval.node),"name",(yyvsp[-4].str));
		    appendChild((yyval.node),firstChild((yyvsp[-1].node)));
		  }
                }
#line 6431 "CParse/parser.c"
    break;

  case 114: /* c_declaration: cpp_lambda_decl  */
#line 3128 "./CParse/parser.y"
                                  {
		  (yyval.node) = (yyvsp[0].node);
		  SWIG_WARN_NODE_BEGIN((yyval.node));
		  Swig_warning(WARN_CPP11_LAMBDA, cparse_file, cparse_line, "Lambda expressions and closures are not fully supported yet.\n");
		  SWIG_WARN_NODE_END((yyval.node));
		}
#line 6442 "CParse/parser.c"
    break;

  case 115: /* c_declaration: USING idcolon EQUAL type plain_declarator SEMI  */
#line 3134 "./CParse/parser.y"
                                                                 {
		  /* Convert using statement to a typedef statement */
		  (yyval.node) = new_node("cdecl");
		  Setattr((yyval.node),"type",(yyvsp[-2].type));
		  Setattr((yyval.node),"storage","typedef");
		  Setattr((yyval.node),"name",(yyvsp[-4].str));
		  Setattr((yyval.node),"decl",(yyvsp[-1].decl).type);
		  SetFlag((yyval.node),"typealias");
		  add_symbols((yyval.node));
		}
#line 6457 "CParse/parser.c"
    break;

  case 116: /* c_declaration: TEMPLATE LESSTHAN template_parms GREATERTHAN USING idcolon EQUAL type plain_declarator SEMI  */
#line 3144 "./CParse/parser.y"
                                                                                                              {
		  /* Convert alias template to a "template" typedef statement */
		  (yyval.node) = new_node("template");
		  Setattr((yyval.node),"type",(yyvsp[-2].type));
		  Setattr((yyval.node),"storage","typedef");
		  Setattr((yyval.node),"name",(yyvsp[-4].str));
		  Setattr((yyval.node),"decl",(yyvsp[-1].decl).type);
		  Setattr((yyval.node),"templateparms",(yyvsp[-7].tparms));
		  Setattr((yyval.node),"templatetype","cdecl");
		  SetFlag((yyval.node),"aliastemplate");
		  add_symbols((yyval.node));
		}
#line 6474 "CParse/parser.c"
    break;

  case 118: /* c_decl: storage_class type declarator cpp_const initializer c_decl_tail  */
#line 3163 "./CParse/parser.y"
                                                                          {
	      String *decl = (yyvsp[-3].decl).type;
              (yyval.node) = new_node("cdecl");
	      if ((yyvsp[-2].dtype).qualifier)
	        decl = add_qualifier_to_declarator((yyvsp[-3].decl).type, (yyvsp[-2].dtype).qualifier);
	      Setattr((yyval.node),"refqualifier",(yyvsp[-2].dtype).refqualifier);
	      Setattr((yyval.node),"type",(yyvsp[-4].type));
	      Setattr((yyval.node),"storage",(yyvsp[-5].str));
	      Setattr((yyval.node),"name",(yyvsp[-3].decl).id);
	      Setattr((yyval.node),"decl",decl);
	      Setattr((yyval.node),"parms",(yyvsp[-3].decl).parms);
	      Setattr((yyval.node),"value",(yyvsp[-1].dtype).val);
	      Setattr((yyval.node),"throws",(yyvsp[-2].dtype).throws);
	      Setattr((yyval.node),"throw",(yyvsp[-2].dtype).throwf);
	      Setattr((yyval.node),"noexcept",(yyvsp[-2].dtype).nexcept);
	      Setattr((yyval.node),"final",(yyvsp[-2].dtype).final);
	      if ((yyvsp[-1].dtype).val && (yyvsp[-1].dtype).type) {
		/* store initializer type as it might be different to the declared type */
		SwigType *valuetype = NewSwigType((yyvsp[-1].dtype).type);
		if (Len(valuetype) > 0) {
		  Setattr((yyval.node), "valuetype", valuetype);
		} else {
		  /* If we can't determine the initializer type use the declared type. */
		  Setattr((yyval.node), "valuetype", (yyvsp[-4].type));
		}
		Delete(valuetype);
	      }
	      if (!(yyvsp[0].node)) {
		if (Len(scanner_ccode)) {
		  String *code = Copy(scanner_ccode);
		  Setattr((yyval.node),"code",code);
		  Delete(code);
		}
	      } else {
		Node *n = (yyvsp[0].node);
		/* Inherit attributes */
		while (n) {
		  String *type = Copy((yyvsp[-4].type));
		  Setattr(n,"type",type);
		  Setattr(n,"storage",(yyvsp[-5].str));
		  n = nextSibling(n);
		  Delete(type);
		}
	      }
	      if ((yyvsp[-1].dtype).bitfield) {
		Setattr((yyval.node),"bitfield", (yyvsp[-1].dtype).bitfield);
	      }

	      if ((yyvsp[-3].decl).id) {
		/* Ignore all scoped declarations, could be 1. out of class function definition 2. friend function declaration 3. ... */
		String *p = Swig_scopename_prefix((yyvsp[-3].decl).id);
		if (p) {
		  /* This is a special case. If the scope name of the declaration exactly
		     matches that of the declaration, then we will allow it. Otherwise, delete. */
		  if ((Namespaceprefix && Strcmp(p, Namespaceprefix) == 0) ||
		      (Classprefix && Strcmp(p, Classprefix) == 0)) {
		    String *lstr = Swig_scopename_last((yyvsp[-3].decl).id);
		    Setattr((yyval.node), "name", lstr);
		    Delete(lstr);
		    set_nextSibling((yyval.node), (yyvsp[0].node));
		  } else {
		    Delete((yyval.node));
		    (yyval.node) = (yyvsp[0].node);
		  }
		  Delete(p);
		} else if (Strncmp((yyvsp[-3].decl).id, "::", 2) == 0) {
		  /* global scope declaration/definition ignored */
		  Delete((yyval.node));
		  (yyval.node) = (yyvsp[0].node);
		} else {
		  set_nextSibling((yyval.node), (yyvsp[0].node));
		}
	      } else {
		Swig_error(cparse_file, cparse_line, "Missing symbol name for global declaration\n");
		(yyval.node) = 0;
	      }

	      if ((yyvsp[-2].dtype).qualifier && (yyvsp[-5].str) && Strstr((yyvsp[-5].str), "static"))
		Swig_error(cparse_file, cparse_line, "Static function %s cannot have a qualifier.\n", Swig_name_decl((yyval.node)));
           }
#line 6559 "CParse/parser.c"
    break;

  case 119: /* c_decl: storage_class type declarator cpp_const EQUAL error SEMI  */
#line 3243 "./CParse/parser.y"
                                                                      {
	      String *decl = (yyvsp[-4].decl).type;
	      (yyval.node) = new_node("cdecl");
	      if ((yyvsp[-3].dtype).qualifier)
	        decl = add_qualifier_to_declarator((yyvsp[-4].decl).type, (yyvsp[-3].dtype).qualifier);
	      Setattr((yyval.node), "refqualifier", (yyvsp[-3].dtype).refqualifier);
	      Setattr((yyval.node), "type", (yyvsp[-5].type));
	      Setattr((yyval.node), "storage", (yyvsp[-6].str));
	      Setattr((yyval.node), "name", (yyvsp[-4].decl).id);
	      Setattr((yyval.node), "decl", decl);
	      Setattr((yyval.node), "parms", (yyvsp[-4].decl).parms);

	      /* Set dummy value to avoid adding in code for handling missing value in later stages */
	      Setattr((yyval.node), "value", "*parse error*");
	      SetFlag((yyval.node), "valueignored");

	      Setattr((yyval.node), "throws", (yyvsp[-3].dtype).throws);
	      Setattr((yyval.node), "throw", (yyvsp[-3].dtype).throwf);
	      Setattr((yyval.node), "noexcept", (yyvsp[-3].dtype).nexcept);
	      Setattr((yyval.node), "final", (yyvsp[-3].dtype).final);

	      if ((yyvsp[-4].decl).id) {
		/* Ignore all scoped declarations, could be 1. out of class function definition 2. friend function declaration 3. ... */
		String *p = Swig_scopename_prefix((yyvsp[-4].decl).id);
		if (p) {
		  if ((Namespaceprefix && Strcmp(p, Namespaceprefix) == 0) ||
		      (Classprefix && Strcmp(p, Classprefix) == 0)) {
		    String *lstr = Swig_scopename_last((yyvsp[-4].decl).id);
		    Setattr((yyval.node), "name", lstr);
		    Delete(lstr);
		  } else {
		    Delete((yyval.node));
		    (yyval.node) = 0;
		  }
		  Delete(p);
		} else if (Strncmp((yyvsp[-4].decl).id, "::", 2) == 0) {
		  /* global scope declaration/definition ignored */
		  Delete((yyval.node));
		  (yyval.node) = 0;
		}
	      }

	      if ((yyvsp[-3].dtype).qualifier && (yyvsp[-6].str) && Strstr((yyvsp[-6].str), "static"))
		Swig_error(cparse_file, cparse_line, "Static function %s cannot have a qualifier.\n", Swig_name_decl((yyval.node)));
	   }
#line 6609 "CParse/parser.c"
    break;

  case 120: /* c_decl: storage_class AUTO declarator cpp_const ARROW cpp_alternate_rettype virt_specifier_seq_opt initializer c_decl_tail  */
#line 3290 "./CParse/parser.y"
                                                                                                                                {
              (yyval.node) = new_node("cdecl");
	      if ((yyvsp[-5].dtype).qualifier) SwigType_push((yyvsp[-6].decl).type, (yyvsp[-5].dtype).qualifier);
	      Setattr((yyval.node),"refqualifier",(yyvsp[-5].dtype).refqualifier);
	      Setattr((yyval.node),"type",(yyvsp[-3].type));
	      Setattr((yyval.node),"storage",(yyvsp[-8].str));
	      Setattr((yyval.node),"name",(yyvsp[-6].decl).id);
	      Setattr((yyval.node),"decl",(yyvsp[-6].decl).type);
	      Setattr((yyval.node),"parms",(yyvsp[-6].decl).parms);
	      Setattr((yyval.node),"throws",(yyvsp[-5].dtype).throws);
	      Setattr((yyval.node),"throw",(yyvsp[-5].dtype).throwf);
	      Setattr((yyval.node),"noexcept",(yyvsp[-5].dtype).nexcept);
	      Setattr((yyval.node),"final",(yyvsp[-5].dtype).final);
	      if (!(yyvsp[0].node)) {
		if (Len(scanner_ccode)) {
		  String *code = Copy(scanner_ccode);
		  Setattr((yyval.node),"code",code);
		  Delete(code);
		}
	      } else {
		Node *n = (yyvsp[0].node);
		while (n) {
		  String *type = Copy((yyvsp[-3].type));
		  Setattr(n,"type",type);
		  Setattr(n,"storage",(yyvsp[-8].str));
		  n = nextSibling(n);
		  Delete(type);
		}
	      }

	      if ((yyvsp[-6].decl).id) {
		/* Ignore all scoped declarations, could be 1. out of class function definition 2. friend function declaration 3. ... */
		String *p = Swig_scopename_prefix((yyvsp[-6].decl).id);
		if (p) {
		  if ((Namespaceprefix && Strcmp(p, Namespaceprefix) == 0) ||
		      (Classprefix && Strcmp(p, Classprefix) == 0)) {
		    String *lstr = Swig_scopename_last((yyvsp[-6].decl).id);
		    Setattr((yyval.node),"name",lstr);
		    Delete(lstr);
		    set_nextSibling((yyval.node), (yyvsp[0].node));
		  } else {
		    Delete((yyval.node));
		    (yyval.node) = (yyvsp[0].node);
		  }
		  Delete(p);
		} else if (Strncmp((yyvsp[-6].decl).id, "::", 2) == 0) {
		  /* global scope declaration/definition ignored */
		  Delete((yyval.node));
		  (yyval.node) = (yyvsp[0].node);
		}
	      } else {
		set_nextSibling((yyval.node), (yyvsp[0].node));
	      }

	      if ((yyvsp[-5].dtype).qualifier && (yyvsp[-8].str) && Strstr((yyvsp[-8].str), "static"))
		Swig_error(cparse_file, cparse_line, "Static function %s cannot have a qualifier.\n", Swig_name_decl((yyval.node)));
           }
#line 6671 "CParse/parser.c"
    break;

  case 121: /* c_decl: storage_class AUTO declarator cpp_const LBRACE  */
#line 3354 "./CParse/parser.y"
                                                            {
	      if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);

              (yyval.node) = new_node("cdecl");
	      if ((yyvsp[-1].dtype).qualifier) SwigType_push((yyvsp[-2].decl).type, (yyvsp[-1].dtype).qualifier);
	      Setattr((yyval.node), "refqualifier", (yyvsp[-1].dtype).refqualifier);
	      Setattr((yyval.node), "type", NewString("auto"));
	      Setattr((yyval.node), "storage", (yyvsp[-4].str));
	      Setattr((yyval.node), "name", (yyvsp[-2].decl).id);
	      Setattr((yyval.node), "decl", (yyvsp[-2].decl).type);
	      Setattr((yyval.node), "parms", (yyvsp[-2].decl).parms);
	      Setattr((yyval.node), "throws", (yyvsp[-1].dtype).throws);
	      Setattr((yyval.node), "throw", (yyvsp[-1].dtype).throwf);
	      Setattr((yyval.node), "noexcept", (yyvsp[-1].dtype).nexcept);
	      Setattr((yyval.node), "final", (yyvsp[-1].dtype).final);

	      if ((yyvsp[-2].decl).id) {
		/* Ignore all scoped declarations, could be 1. out of class function definition 2. friend function declaration 3. ... */
		String *p = Swig_scopename_prefix((yyvsp[-2].decl).id);
		if (p) {
		  if ((Namespaceprefix && Strcmp(p, Namespaceprefix) == 0) ||
		      (Classprefix && Strcmp(p, Classprefix) == 0)) {
		    String *lstr = Swig_scopename_last((yyvsp[-2].decl).id);
		    Setattr((yyval.node), "name", lstr);
		    Delete(lstr);
		  } else {
		    Delete((yyval.node));
		    (yyval.node) = 0;
		  }
		  Delete(p);
		} else if (Strncmp((yyvsp[-2].decl).id, "::", 2) == 0) {
		  /* global scope declaration/definition ignored */
		  Delete((yyval.node));
		  (yyval.node) = 0;
		}
	      }

	      if ((yyvsp[-1].dtype).qualifier && (yyvsp[-4].str) && Strstr((yyvsp[-4].str), "static"))
		Swig_error(cparse_file, cparse_line, "Static function %s cannot have a qualifier.\n", Swig_name_decl((yyval.node)));
	   }
#line 6716 "CParse/parser.c"
    break;

  case 122: /* c_decl: storage_class AUTO idcolon EQUAL definetype SEMI  */
#line 3395 "./CParse/parser.y"
                                                              {
	      SwigType *type = deduce_type(&(yyvsp[-1].dtype));
	      if (!type)
		type = NewString("auto");
	      (yyval.node) = new_node("cdecl");
	      Setattr((yyval.node), "type", type);
	      Setattr((yyval.node), "storage", (yyvsp[-5].str));
	      Setattr((yyval.node), "name", (yyvsp[-3].str));
	      Setattr((yyval.node), "decl", NewStringEmpty());
	      Setattr((yyval.node), "value", (yyvsp[-1].dtype).val);
	      Setattr((yyval.node), "valuetype", type);
	   }
#line 6733 "CParse/parser.c"
    break;

  case 123: /* c_decl_tail: SEMI  */
#line 3411 "./CParse/parser.y"
                      { 
                   (yyval.node) = 0;
                   Clear(scanner_ccode); 
               }
#line 6742 "CParse/parser.c"
    break;

  case 124: /* c_decl_tail: COMMA declarator cpp_const initializer c_decl_tail  */
#line 3415 "./CParse/parser.y"
                                                                    {
		 (yyval.node) = new_node("cdecl");
		 if ((yyvsp[-2].dtype).qualifier) SwigType_push((yyvsp[-3].decl).type,(yyvsp[-2].dtype).qualifier);
		 Setattr((yyval.node),"refqualifier",(yyvsp[-2].dtype).refqualifier);
		 Setattr((yyval.node),"name",(yyvsp[-3].decl).id);
		 Setattr((yyval.node),"decl",(yyvsp[-3].decl).type);
		 Setattr((yyval.node),"parms",(yyvsp[-3].decl).parms);
		 Setattr((yyval.node),"value",(yyvsp[-1].dtype).val);
		 Setattr((yyval.node),"throws",(yyvsp[-2].dtype).throws);
		 Setattr((yyval.node),"throw",(yyvsp[-2].dtype).throwf);
		 Setattr((yyval.node),"noexcept",(yyvsp[-2].dtype).nexcept);
		 Setattr((yyval.node),"final",(yyvsp[-2].dtype).final);
		 if ((yyvsp[-1].dtype).bitfield) {
		   Setattr((yyval.node),"bitfield", (yyvsp[-1].dtype).bitfield);
		 }
		 if (!(yyvsp[0].node)) {
		   if (Len(scanner_ccode)) {
		     String *code = Copy(scanner_ccode);
		     Setattr((yyval.node),"code",code);
		     Delete(code);
		   }
		 } else {
		   set_nextSibling((yyval.node), (yyvsp[0].node));
		 }
	       }
#line 6772 "CParse/parser.c"
    break;

  case 125: /* c_decl_tail: LBRACE  */
#line 3440 "./CParse/parser.y"
                        { 
                   if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
                   (yyval.node) = 0;
               }
#line 6781 "CParse/parser.c"
    break;

  case 126: /* c_decl_tail: error  */
#line 3444 "./CParse/parser.y"
                       {
		   (yyval.node) = 0;
		   if (yychar == RPAREN) {
		       Swig_error(cparse_file, cparse_line, "Unexpected closing parenthesis (')').\n");
		   } else {
		       Swig_error(cparse_file, cparse_line, "Syntax error - possibly a missing semicolon (';').\n");
		   }
		   Exit(EXIT_FAILURE);
               }
#line 6795 "CParse/parser.c"
    break;

  case 132: /* cpp_alternate_rettype: idcolon  */
#line 3462 "./CParse/parser.y"
                        { (yyval.type) = (yyvsp[0].str); }
#line 6801 "CParse/parser.c"
    break;

  case 133: /* cpp_alternate_rettype: idcolon AND  */
#line 3463 "./CParse/parser.y"
                            {
                (yyval.type) = (yyvsp[-1].str);
                SwigType_add_reference((yyval.type));
              }
#line 6810 "CParse/parser.c"
    break;

  case 135: /* cpp_lambda_decl: storage_class AUTO idcolon EQUAL lambda_introducer lambda_template LPAREN parms RPAREN cpp_const lambda_body lambda_tail  */
#line 3478 "./CParse/parser.y"
                                                                                                                                           {
		  (yyval.node) = new_node("lambda");
		  Setattr((yyval.node),"name",(yyvsp[-9].str));
		  Delete((yyvsp[-11].str));
		  add_symbols((yyval.node));
	        }
#line 6821 "CParse/parser.c"
    break;

  case 136: /* cpp_lambda_decl: storage_class AUTO idcolon EQUAL lambda_introducer lambda_template LPAREN parms RPAREN cpp_const ARROW type lambda_body lambda_tail  */
#line 3484 "./CParse/parser.y"
                                                                                                                                                      {
		  (yyval.node) = new_node("lambda");
		  Setattr((yyval.node),"name",(yyvsp[-11].str));
		  Delete((yyvsp[-13].str));
		  add_symbols((yyval.node));
		}
#line 6832 "CParse/parser.c"
    break;

  case 137: /* cpp_lambda_decl: storage_class AUTO idcolon EQUAL lambda_introducer lambda_template lambda_body lambda_tail  */
#line 3490 "./CParse/parser.y"
                                                                                                             {
		  (yyval.node) = new_node("lambda");
		  Setattr((yyval.node),"name",(yyvsp[-5].str));
		  Delete((yyvsp[-7].str));
		  add_symbols((yyval.node));
		}
#line 6843 "CParse/parser.c"
    break;

  case 138: /* lambda_introducer: LBRACKET  */
#line 3498 "./CParse/parser.y"
                             {
		  if (skip_balanced('[',']') < 0) Exit(EXIT_FAILURE);
		  (yyval.node) = 0;
	        }
#line 6852 "CParse/parser.c"
    break;

  case 139: /* lambda_template: LESSTHAN  */
#line 3504 "./CParse/parser.y"
                           {
		  if (skip_balanced('<','>') < 0) Exit(EXIT_FAILURE);
		  (yyval.node) = 0;
		}
#line 6861 "CParse/parser.c"
    break;

  case 140: /* lambda_template: %empty  */
#line 3508 "./CParse/parser.y"
                         { (yyval.node) = 0; }
#line 6867 "CParse/parser.c"
    break;

  case 141: /* lambda_body: LBRACE  */
#line 3511 "./CParse/parser.y"
                     {
		  if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		  (yyval.node) = 0;
		}
#line 6876 "CParse/parser.c"
    break;

  case 142: /* lambda_tail: SEMI  */
#line 3516 "./CParse/parser.y"
                     {
		  (yyval.pl) = 0;
		}
#line 6884 "CParse/parser.c"
    break;

  case 143: /* $@4: %empty  */
#line 3519 "./CParse/parser.y"
                         {
		  if (skip_balanced('(',')') < 0) Exit(EXIT_FAILURE);
		}
#line 6892 "CParse/parser.c"
    break;

  case 144: /* lambda_tail: LPAREN $@4 SEMI  */
#line 3521 "./CParse/parser.y"
                       {
		  (yyval.pl) = 0;
		}
#line 6900 "CParse/parser.c"
    break;

  case 145: /* c_enum_key: ENUM  */
#line 3532 "./CParse/parser.y"
                  {
		   (yyval.id) = "enum";
	      }
#line 6908 "CParse/parser.c"
    break;

  case 146: /* c_enum_key: ENUM CLASS  */
#line 3535 "./CParse/parser.y"
                           {
		   (yyval.id) = "enum class";
	      }
#line 6916 "CParse/parser.c"
    break;

  case 147: /* c_enum_key: ENUM STRUCT  */
#line 3538 "./CParse/parser.y"
                            {
		   (yyval.id) = "enum struct";
	      }
#line 6924 "CParse/parser.c"
    break;

  case 148: /* c_enum_inherit: COLON type_right  */
#line 3547 "./CParse/parser.y"
                                  {
                   (yyval.node) = (yyvsp[0].type);
              }
#line 6932 "CParse/parser.c"
    break;

  case 149: /* c_enum_inherit: %empty  */
#line 3550 "./CParse/parser.y"
                       { (yyval.node) = 0; }
#line 6938 "CParse/parser.c"
    break;

  case 150: /* c_enum_forward_decl: storage_class c_enum_key ename c_enum_inherit SEMI  */
#line 3557 "./CParse/parser.y"
                                                                         {
		   SwigType *ty = 0;
		   int scopedenum = (yyvsp[-2].id) && !Equal((yyvsp[-3].id), "enum");
		   (yyval.node) = new_node("enumforward");
		   ty = NewStringf("enum %s", (yyvsp[-2].id));
		   Setattr((yyval.node),"enumkey",(yyvsp[-3].id));
		   if (scopedenum)
		     SetFlag((yyval.node), "scopedenum");
		   Setattr((yyval.node),"name",(yyvsp[-2].id));
		   Setattr((yyval.node), "enumbase", (yyvsp[-1].node));
		   Setattr((yyval.node),"type",ty);
		   Setattr((yyval.node),"sym:weak", "1");
		   add_symbols((yyval.node));
	      }
#line 6957 "CParse/parser.c"
    break;

  case 151: /* c_enum_decl: storage_class c_enum_key ename c_enum_inherit LBRACE enumlist RBRACE SEMI  */
#line 3579 "./CParse/parser.y"
                                                                                         {
		  SwigType *ty = 0;
		  int scopedenum = (yyvsp[-5].id) && !Equal((yyvsp[-6].id), "enum");
                  (yyval.node) = new_node("enum");
		  ty = NewStringf("enum %s", (yyvsp[-5].id));
		  Setattr((yyval.node),"enumkey",(yyvsp[-6].id));
		  if (scopedenum)
		    SetFlag((yyval.node), "scopedenum");
		  Setattr((yyval.node),"name",(yyvsp[-5].id));
		  Setattr((yyval.node), "enumbase", (yyvsp[-4].node));
		  Setattr((yyval.node),"type",ty);
		  appendChild((yyval.node),(yyvsp[-2].node));
		  add_symbols((yyval.node));      /* Add to tag space */

		  if (scopedenum) {
		    Swig_symbol_newscope();
		    Swig_symbol_setscopename((yyvsp[-5].id));
		    Delete(Namespaceprefix);
		    Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		  }

		  add_symbols((yyvsp[-2].node));      /* Add enum values to appropriate enum or enum class scope */

		  if (scopedenum) {
		    Setattr((yyval.node),"symtab", Swig_symbol_popscope());
		    Delete(Namespaceprefix);
		    Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		  }
               }
#line 6991 "CParse/parser.c"
    break;

  case 152: /* c_enum_decl: storage_class c_enum_key ename c_enum_inherit LBRACE enumlist RBRACE declarator cpp_const initializer c_decl_tail  */
#line 3608 "./CParse/parser.y"
                                                                                                                                   {
		 Node *n;
		 SwigType *ty = 0;
		 String   *unnamed = 0;
		 int       unnamedinstance = 0;
		 int scopedenum = (yyvsp[-8].id) && !Equal((yyvsp[-9].id), "enum");

		 (yyval.node) = new_node("enum");
		 Setattr((yyval.node),"enumkey",(yyvsp[-9].id));
		 if (scopedenum)
		   SetFlag((yyval.node), "scopedenum");
		 Setattr((yyval.node), "enumbase", (yyvsp[-7].node));
		 if ((yyvsp[-8].id)) {
		   Setattr((yyval.node),"name",(yyvsp[-8].id));
		   ty = NewStringf("enum %s", (yyvsp[-8].id));
		 } else if ((yyvsp[-3].decl).id) {
		   unnamed = make_unnamed();
		   ty = NewStringf("enum %s", unnamed);
		   Setattr((yyval.node),"unnamed",unnamed);
                   /* name is not set for unnamed enum instances, e.g. enum { foo } Instance; */
		   if ((yyvsp[-10].str) && Cmp((yyvsp[-10].str),"typedef") == 0) {
		     Setattr((yyval.node),"name",(yyvsp[-3].decl).id);
                   } else {
                     unnamedinstance = 1;
                   }
		   Setattr((yyval.node),"storage",(yyvsp[-10].str));
		 }
		 if ((yyvsp[-3].decl).id && Cmp((yyvsp[-10].str),"typedef") == 0) {
		   Setattr((yyval.node),"tdname",(yyvsp[-3].decl).id);
                   Setattr((yyval.node),"allows_typedef","1");
                 }
		 appendChild((yyval.node),(yyvsp[-5].node));
		 n = new_node("cdecl");
		 Setattr(n,"type",ty);
		 Setattr(n,"name",(yyvsp[-3].decl).id);
		 Setattr(n,"storage",(yyvsp[-10].str));
		 Setattr(n,"decl",(yyvsp[-3].decl).type);
		 Setattr(n,"parms",(yyvsp[-3].decl).parms);
		 Setattr(n,"unnamed",unnamed);

                 if (unnamedinstance) {
		   SwigType *cty = NewString("enum ");
		   Setattr((yyval.node),"type",cty);
		   SetFlag((yyval.node),"unnamedinstance");
		   SetFlag(n,"unnamedinstance");
		   Delete(cty);
                 }
		 if ((yyvsp[0].node)) {
		   Node *p = (yyvsp[0].node);
		   set_nextSibling(n,p);
		   while (p) {
		     SwigType *cty = Copy(ty);
		     Setattr(p,"type",cty);
		     Setattr(p,"unnamed",unnamed);
		     Setattr(p,"storage",(yyvsp[-10].str));
		     Delete(cty);
		     p = nextSibling(p);
		   }
		 } else {
		   if (Len(scanner_ccode)) {
		     String *code = Copy(scanner_ccode);
		     Setattr(n,"code",code);
		     Delete(code);
		   }
		 }

                 /* Ensure that typedef enum ABC {foo} XYZ; uses XYZ for sym:name, like structs.
                  * Note that class_rename/yyrename are bit of a mess so used this simple approach to change the name. */
                 if ((yyvsp[-3].decl).id && (yyvsp[-8].id) && Cmp((yyvsp[-10].str),"typedef") == 0) {
		   String *name = NewString((yyvsp[-3].decl).id);
                   Setattr((yyval.node), "parser:makename", name);
		   Delete(name);
                 }

		 add_symbols((yyval.node));       /* Add enum to tag space */
		 set_nextSibling((yyval.node),n);
		 Delete(n);

		 if (scopedenum) {
		   Swig_symbol_newscope();
		   Swig_symbol_setscopename((yyvsp[-8].id));
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		 }

		 add_symbols((yyvsp[-5].node));      /* Add enum values to appropriate enum or enum class scope */

		 if (scopedenum) {
		   Setattr((yyval.node),"symtab", Swig_symbol_popscope());
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		 }

	         add_symbols(n);
		 Delete(unnamed);
	       }
#line 7092 "CParse/parser.c"
    break;

  case 153: /* c_constructor_decl: storage_class type LPAREN parms RPAREN ctor_end  */
#line 3706 "./CParse/parser.y"
                                                                     {
                   /* This is a sick hack.  If the ctor_end has parameters,
                      and the parms parameter only has 1 parameter, this
                      could be a declaration of the form:

                         type (id)(parms)

			 Otherwise it's an error. */
                    int err = 0;
                    (yyval.node) = 0;

		    if ((ParmList_len((yyvsp[-2].pl)) == 1) && (!Swig_scopename_check((yyvsp[-4].type)))) {
		      SwigType *ty = Getattr((yyvsp[-2].pl),"type");
		      String *name = Getattr((yyvsp[-2].pl),"name");
		      err = 1;
		      if (!name) {
			(yyval.node) = new_node("cdecl");
			Setattr((yyval.node),"type",(yyvsp[-4].type));
			Setattr((yyval.node),"storage",(yyvsp[-5].str));
			Setattr((yyval.node),"name",ty);

			if ((yyvsp[0].decl).have_parms) {
			  SwigType *decl = NewStringEmpty();
			  SwigType_add_function(decl,(yyvsp[0].decl).parms);
			  Setattr((yyval.node),"decl",decl);
			  Setattr((yyval.node),"parms",(yyvsp[0].decl).parms);
			  if (Len(scanner_ccode)) {
			    String *code = Copy(scanner_ccode);
			    Setattr((yyval.node),"code",code);
			    Delete(code);
			  }
			}
			if ((yyvsp[0].decl).defarg) {
			  Setattr((yyval.node),"value",(yyvsp[0].decl).defarg);
			}
			Setattr((yyval.node),"throws",(yyvsp[0].decl).throws);
			Setattr((yyval.node),"throw",(yyvsp[0].decl).throwf);
			Setattr((yyval.node),"noexcept",(yyvsp[0].decl).nexcept);
			Setattr((yyval.node),"final",(yyvsp[0].decl).final);
			err = 0;
		      }
		    } else {
		      Delete((yyvsp[-5].str));
		    }
		    if (err) {
		      Swig_error(cparse_file,cparse_line,"Syntax error in input(2).\n");
		      Exit(EXIT_FAILURE);
		    }
                }
#line 7146 "CParse/parser.c"
    break;

  case 160: /* @5: %empty  */
#line 3775 "./CParse/parser.y"
                                                                                            {
                   String *prefix;
                   List *bases = 0;
		   Node *scope = 0;
		   int errored_flag = 0;
		   String *code;
		   (yyval.node) = new_node("class");
		   Setattr((yyval.node),"kind",(yyvsp[-4].type));
		   if ((yyvsp[-1].bases)) {
		     Setattr((yyval.node),"baselist", Getattr((yyvsp[-1].bases),"public"));
		     Setattr((yyval.node),"protectedbaselist", Getattr((yyvsp[-1].bases),"protected"));
		     Setattr((yyval.node),"privatebaselist", Getattr((yyvsp[-1].bases),"private"));
		   }
		   Setattr((yyval.node),"allows_typedef","1");

		   /* preserve the current scope */
		   Setattr((yyval.node),"prev_symtab",Swig_symbol_current());
		  
		   /* If the class name is qualified.  We need to create or lookup namespace/scope entries */
		   scope = resolve_create_node_scope((yyvsp[-3].str), 1, &errored_flag);
		   /* save nscope_inner to the class - it may be overwritten in nested classes*/
		   Setattr((yyval.node), "nested:innerscope", nscope_inner);
		   Setattr((yyval.node), "nested:nscope", nscope);
		   Setfile(scope,cparse_file);
		   Setline(scope,cparse_line);
		   Setattr((yyval.node), "name", scope);

		   if (currentOuterClass) {
		     SetFlag((yyval.node), "nested");
		     Setattr((yyval.node), "nested:outer", currentOuterClass);
		     set_access_mode((yyval.node));
		   }
		   Swig_features_get(Swig_cparse_features(), Namespaceprefix, Getattr((yyval.node), "name"), 0, (yyval.node));
		   /* save yyrename to the class attribute, to be used later in add_symbols()*/
		   Setattr((yyval.node), "class_rename", make_name((yyval.node), scope, 0));
		   Setattr((yyval.node), "Classprefix", scope);
		   Classprefix = NewString(scope);
		   /* Deal with inheritance  */
		   if ((yyvsp[-1].bases))
		     bases = Swig_make_inherit_list(scope, Getattr((yyvsp[-1].bases), "public"), Namespaceprefix);
		   prefix = SwigType_istemplate_templateprefix(scope);
		   if (prefix) {
		     String *fbase, *tbase;
		     if (Namespaceprefix) {
		       fbase = NewStringf("%s::%s", Namespaceprefix, scope);
		       tbase = NewStringf("%s::%s", Namespaceprefix, prefix);
		     } else {
		       fbase = Copy(scope);
		       tbase = Copy(prefix);
		     }
		     Swig_name_inherit(tbase,fbase);
		     Delete(fbase);
		     Delete(tbase);
		   }
                   if (Strcmp((yyvsp[-4].type), "class") == 0) {
		     cplus_mode = CPLUS_PRIVATE;
		   } else {
		     cplus_mode = CPLUS_PUBLIC;
		   }
		   if (!cparse_cplusplus) {
		     set_scope_to_global();
		   }
		   Swig_symbol_newscope();
		   Swig_symbol_setscopename(scope);
		   Swig_inherit_base_symbols(bases);
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   cparse_start_line = cparse_line;

		   /* If there are active template parameters, we need to make sure they are
                      placed in the class symbol table so we can catch shadows */

		   if (template_parameters) {
		     Parm *tp = template_parameters;
		     while(tp) {
		       String *tpname = Copy(Getattr(tp,"name"));
		       Node *tn = new_node("templateparm");
		       Setattr(tn,"name",tpname);
		       Swig_symbol_cadd(tpname,tn);
		       tp = nextSibling(tp);
		       Delete(tpname);
		     }
		   }
		   Delete(prefix);
		   inclass = 1;
		   currentOuterClass = (yyval.node);
		   if (cparse_cplusplusout) {
		     /* save the structure declaration to declare it in global scope for C++ to see */
		     code = get_raw_text_balanced('{', '}');
		     Setattr((yyval.node), "code", code);
		     Delete(code);
		   }
               }
#line 7244 "CParse/parser.c"
    break;

  case 161: /* cpp_class_decl: storage_class cpptype idcolon class_virt_specifier_opt inherit LBRACE @5 cpp_members RBRACE cpp_opt_declarators  */
#line 3867 "./CParse/parser.y"
                                                        {
		   Node *p;
		   SwigType *ty;
		   Symtab *cscope;
		   Node *am = 0;
		   String *scpname = 0;
		   (yyval.node) = currentOuterClass;
		   currentOuterClass = Getattr((yyval.node), "nested:outer");
		   nscope_inner = Getattr((yyval.node), "nested:innerscope");
		   nscope = Getattr((yyval.node), "nested:nscope");
		   Delattr((yyval.node), "nested:innerscope");
		   Delattr((yyval.node), "nested:nscope");
		   if (nscope_inner && Strcmp(nodeType(nscope_inner), "class") == 0) { /* actual parent class for this class */
		     Node* forward_declaration = Swig_symbol_clookup_no_inherit(Getattr((yyval.node),"name"), Getattr(nscope_inner, "symtab"));
		     if (forward_declaration) {
		       Setattr((yyval.node), "access", Getattr(forward_declaration, "access"));
		     }
		     Setattr((yyval.node), "nested:outer", nscope_inner);
		     SetFlag((yyval.node), "nested");
                   }
		   if (!currentOuterClass)
		     inclass = 0;
		   cscope = Getattr((yyval.node), "prev_symtab");
		   Delattr((yyval.node), "prev_symtab");
		   
		   /* Check for pure-abstract class */
		   Setattr((yyval.node),"abstracts", pure_abstracts((yyvsp[-2].node)));
		   
		   /* This bit of code merges in a previously defined %extend directive (if any) */
		   {
		     String *clsname = Swig_symbol_qualifiedscopename(0);
		     am = Getattr(Swig_extend_hash(), clsname);
		     if (am) {
		       Swig_extend_merge((yyval.node), am);
		       Delattr(Swig_extend_hash(), clsname);
		     }
		     Delete(clsname);
		   }
		   if (!classes) classes = NewHash();
		   scpname = Swig_symbol_qualifiedscopename(0);
		   Setattr(classes, scpname, (yyval.node));

		   appendChild((yyval.node), (yyvsp[-2].node));
		   
		   if (am) 
		     Swig_extend_append_previous((yyval.node), am);

		   p = (yyvsp[0].node);
		   if (p && !nscope_inner) {
		     if (!cparse_cplusplus && currentOuterClass)
		       appendChild(currentOuterClass, p);
		     else
		      appendSibling((yyval.node), p);
		   }
		   
		   if (nscope_inner) {
		     ty = NewString(scpname); /* if the class is declared out of scope, let the declarator use fully qualified type*/
		   } else if (cparse_cplusplus && !cparse_externc) {
		     ty = NewString(Getattr((yyvsp[-3].node), "name"));
		   } else {
		     ty = NewStringf("%s %s", (yyvsp[-8].type), Getattr((yyvsp[-3].node), "name"));
		   }
		   while (p) {
		     Setattr(p, "storage", (yyvsp[-9].str));
		     Setattr(p, "type" ,ty);
		     if (!cparse_cplusplus && currentOuterClass && (!Getattr(currentOuterClass, "name"))) {
		       SetFlag(p, "hasconsttype");
		     }
		     p = nextSibling(p);
		   }
		   if ((yyvsp[0].node) && Cmp((yyvsp[-9].str),"typedef") == 0)
		     add_typedef_name((yyval.node), (yyvsp[0].node), Getattr((yyvsp[-3].node), "name"), cscope, scpname);
		   Delete(scpname);

		   if (cplus_mode != CPLUS_PUBLIC) {
		   /* we 'open' the class at the end, to allow %template
		      to add new members */
		     Node *pa = new_node("access");
		     Setattr(pa, "kind", "public");
		     cplus_mode = CPLUS_PUBLIC;
		     appendChild((yyval.node), pa);
		     Delete(pa);
		   }
		   if (currentOuterClass)
		     restore_access_mode((yyval.node));
		   Setattr((yyval.node), "symtab", Swig_symbol_popscope());
		   Classprefix = Getattr((yyval.node), "Classprefix");
		   Delattr((yyval.node), "Classprefix");
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   if (cplus_mode == CPLUS_PRIVATE) {
		     (yyval.node) = 0; /* skip private nested classes */
		   } else if (cparse_cplusplus && currentOuterClass && ignore_nested_classes && !GetFlag((yyval.node), "feature:flatnested")) {
		     (yyval.node) = nested_forward_declaration((yyvsp[-9].str), (yyvsp[-8].type), Getattr((yyvsp[-3].node), "name"), Copy(Getattr((yyvsp[-3].node), "name")), (yyvsp[0].node));
		   } else if (nscope_inner) {
		     /* this is tricky */
		     /* we add the declaration in the original namespace */
		     if (Strcmp(nodeType(nscope_inner), "class") == 0 && cparse_cplusplus && ignore_nested_classes && !GetFlag((yyval.node), "feature:flatnested"))
		       (yyval.node) = nested_forward_declaration((yyvsp[-9].str), (yyvsp[-8].type), Getattr((yyvsp[-3].node), "name"), Copy(Getattr((yyvsp[-3].node), "name")), (yyvsp[0].node));
		     appendChild(nscope_inner, (yyval.node));
		     Swig_symbol_setscope(Getattr(nscope_inner, "symtab"));
		     Delete(Namespaceprefix);
		     Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		     yyrename = Copy(Getattr((yyval.node), "class_rename"));
		     add_symbols((yyval.node));
		     Delattr((yyval.node), "class_rename");
		     /* but the variable definition in the current scope */
		     Swig_symbol_setscope(cscope);
		     Delete(Namespaceprefix);
		     Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		     add_symbols((yyvsp[0].node));
		     if (nscope) {
		       (yyval.node) = nscope; /* here we return recreated namespace tower instead of the class itself */
		       if ((yyvsp[0].node)) {
			 appendSibling((yyval.node), (yyvsp[0].node));
		       }
		     } else if (!SwigType_istemplate(ty) && template_parameters == 0) { /* for template we need the class itself */
		       (yyval.node) = (yyvsp[0].node);
		     }
		   } else {
		     Delete(yyrename);
		     yyrename = 0;
		     if (!cparse_cplusplus && currentOuterClass) { /* nested C structs go into global scope*/
		       Node *outer = currentOuterClass;
		       while (Getattr(outer, "nested:outer"))
			 outer = Getattr(outer, "nested:outer");
		       appendSibling(outer, (yyval.node));
		       Swig_symbol_setscope(cscope); /* declaration goes in the parent scope */
		       add_symbols((yyvsp[0].node));
		       set_scope_to_global();
		       Delete(Namespaceprefix);
		       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		       yyrename = Copy(Getattr((yyval.node), "class_rename"));
		       add_symbols((yyval.node));
		       if (!cparse_cplusplusout)
			 Delattr((yyval.node), "nested:outer");
		       Delattr((yyval.node), "class_rename");
		       (yyval.node) = 0;
		     } else {
		       yyrename = Copy(Getattr((yyval.node), "class_rename"));
		       add_symbols((yyval.node));
		       add_symbols((yyvsp[0].node));
		       Delattr((yyval.node), "class_rename");
		     }
		   }
		   Delete(ty);
		   Swig_symbol_setscope(cscope);
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   Classprefix = currentOuterClass ? Getattr(currentOuterClass, "Classprefix") : 0;
	       }
#line 7400 "CParse/parser.c"
    break;

  case 162: /* @6: %empty  */
#line 4021 "./CParse/parser.y"
                                                          {
	       String *unnamed;
	       String *code;
	       unnamed = make_unnamed();
	       (yyval.node) = new_node("class");
	       Setattr((yyval.node),"kind",(yyvsp[-2].type));
	       if ((yyvsp[-1].bases)) {
		 Setattr((yyval.node),"baselist", Getattr((yyvsp[-1].bases),"public"));
		 Setattr((yyval.node),"protectedbaselist", Getattr((yyvsp[-1].bases),"protected"));
		 Setattr((yyval.node),"privatebaselist", Getattr((yyvsp[-1].bases),"private"));
	       }
	       Setattr((yyval.node),"storage",(yyvsp[-3].str));
	       Setattr((yyval.node),"unnamed",unnamed);
	       Setattr((yyval.node),"allows_typedef","1");
	       if (currentOuterClass) {
		 SetFlag((yyval.node), "nested");
		 Setattr((yyval.node), "nested:outer", currentOuterClass);
		 set_access_mode((yyval.node));
	       }
	       Swig_features_get(Swig_cparse_features(), Namespaceprefix, 0, 0, (yyval.node));
	       /* save yyrename to the class attribute, to be used later in add_symbols()*/
	       Setattr((yyval.node), "class_rename", make_name((yyval.node),0,0));
	       if (Strcmp((yyvsp[-2].type), "class") == 0) {
		 cplus_mode = CPLUS_PRIVATE;
	       } else {
		 cplus_mode = CPLUS_PUBLIC;
	       }
	       Swig_symbol_newscope();
	       cparse_start_line = cparse_line;
	       currentOuterClass = (yyval.node);
	       inclass = 1;
	       Classprefix = 0;
	       Delete(Namespaceprefix);
	       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	       /* save the structure declaration to make a typedef for it later*/
	       code = get_raw_text_balanced('{', '}');
	       Setattr((yyval.node), "code", code);
	       Delete(code);
	     }
#line 7444 "CParse/parser.c"
    break;

  case 163: /* cpp_class_decl: storage_class cpptype inherit LBRACE @6 cpp_members RBRACE cpp_opt_declarators  */
#line 4059 "./CParse/parser.y"
                                                      {
	       String *unnamed;
               List *bases = 0;
	       String *name = 0;
	       Node *n;
	       Classprefix = 0;
	       (void)(yyvsp[-3].node);
	       (yyval.node) = currentOuterClass;
	       currentOuterClass = Getattr((yyval.node), "nested:outer");
	       if (!currentOuterClass)
		 inclass = 0;
	       else
		 restore_access_mode((yyval.node));
	       unnamed = Getattr((yyval.node),"unnamed");
               /* Check for pure-abstract class */
	       Setattr((yyval.node),"abstracts", pure_abstracts((yyvsp[-2].node)));
	       n = (yyvsp[0].node);
	       if (cparse_cplusplus && currentOuterClass && ignore_nested_classes && !GetFlag((yyval.node), "feature:flatnested")) {
		 String *name = n ? Copy(Getattr(n, "name")) : 0;
		 (yyval.node) = nested_forward_declaration((yyvsp[-7].str), (yyvsp[-6].type), 0, name, n);
		 Swig_symbol_popscope();
	         Delete(Namespaceprefix);
		 Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	       } else if (n) {
	         appendSibling((yyval.node),n);
		 /* If a proper typedef name was given, we'll use it to set the scope name */
		 name = try_to_find_a_name_for_unnamed_structure((yyvsp[-7].str), n);
		 if (name) {
		   String *scpname = 0;
		   SwigType *ty;
		   Setattr((yyval.node),"tdname",name);
		   Setattr((yyval.node),"name",name);
		   Swig_symbol_setscopename(name);
		   if ((yyvsp[-5].bases))
		     bases = Swig_make_inherit_list(name,Getattr((yyvsp[-5].bases),"public"),Namespaceprefix);
		   Swig_inherit_base_symbols(bases);

		     /* If a proper name was given, we use that as the typedef, not unnamed */
		   Clear(unnamed);
		   Append(unnamed, name);
		   if (cparse_cplusplus && !cparse_externc) {
		     ty = NewString(name);
		   } else {
		     ty = NewStringf("%s %s", (yyvsp[-6].type),name);
		   }
		   while (n) {
		     Setattr(n,"storage",(yyvsp[-7].str));
		     Setattr(n, "type", ty);
		     if (!cparse_cplusplus && currentOuterClass && (!Getattr(currentOuterClass, "name"))) {
		       SetFlag(n,"hasconsttype");
		     }
		     n = nextSibling(n);
		   }
		   n = (yyvsp[0].node);

		   /* Check for previous extensions */
		   {
		     String *clsname = Swig_symbol_qualifiedscopename(0);
		     Node *am = Getattr(Swig_extend_hash(),clsname);
		     if (am) {
		       /* Merge the extension into the symbol table */
		       Swig_extend_merge((yyval.node),am);
		       Swig_extend_append_previous((yyval.node),am);
		       Delattr(Swig_extend_hash(),clsname);
		     }
		     Delete(clsname);
		   }
		   if (!classes) classes = NewHash();
		   scpname = Swig_symbol_qualifiedscopename(0);
		   Setattr(classes,scpname,(yyval.node));
		   Delete(scpname);
		 } else { /* no suitable name was found for a struct */
		   Setattr((yyval.node), "nested:unnamed", Getattr(n, "name")); /* save the name of the first declarator for later use in name generation*/
		   while (n) { /* attach unnamed struct to the declarators, so that they would receive proper type later*/
		     Setattr(n, "nested:unnamedtype", (yyval.node));
		     Setattr(n, "storage", (yyvsp[-7].str));
		     n = nextSibling(n);
		   }
		   n = (yyvsp[0].node);
		   Swig_symbol_setscopename("<unnamed>");
		 }
		 appendChild((yyval.node),(yyvsp[-2].node));
		 /* Pop the scope */
		 Setattr((yyval.node),"symtab",Swig_symbol_popscope());
		 if (name) {
		   Delete(yyrename);
		   yyrename = Copy(Getattr((yyval.node), "class_rename"));
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   add_symbols((yyval.node));
		   add_symbols(n);
		   Delattr((yyval.node), "class_rename");
		 }else if (cparse_cplusplus)
		   (yyval.node) = 0; /* ignore unnamed structs for C++ */
	         Delete(unnamed);
	       } else { /* unnamed struct w/o declarator*/
		 Swig_symbol_popscope();
	         Delete(Namespaceprefix);
		 Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		 add_symbols((yyvsp[-2].node));
		 Delete((yyval.node));
		 (yyval.node) = (yyvsp[-2].node); /* pass member list to outer class/namespace (instead of self)*/
	       }
	       Classprefix = currentOuterClass ? Getattr(currentOuterClass, "Classprefix") : 0;
              }
#line 7554 "CParse/parser.c"
    break;

  case 164: /* cpp_opt_declarators: SEMI  */
#line 4166 "./CParse/parser.y"
                            { (yyval.node) = 0; }
#line 7560 "CParse/parser.c"
    break;

  case 165: /* cpp_opt_declarators: declarator cpp_const initializer c_decl_tail  */
#line 4167 "./CParse/parser.y"
                                                                    {
                        (yyval.node) = new_node("cdecl");
                        Setattr((yyval.node),"name",(yyvsp[-3].decl).id);
                        Setattr((yyval.node),"decl",(yyvsp[-3].decl).type);
                        Setattr((yyval.node),"parms",(yyvsp[-3].decl).parms);
			set_nextSibling((yyval.node), (yyvsp[0].node));
                    }
#line 7572 "CParse/parser.c"
    break;

  case 166: /* cpp_forward_class_decl: storage_class cpptype idcolon SEMI  */
#line 4179 "./CParse/parser.y"
                                                            {
	      if ((yyvsp[-3].str) && Strstr((yyvsp[-3].str), "friend")) {
		/* Ignore */
                (yyval.node) = 0; 
	      } else {
		(yyval.node) = new_node("classforward");
		Setattr((yyval.node),"kind",(yyvsp[-2].type));
		Setattr((yyval.node),"name",(yyvsp[-1].str));
		Setattr((yyval.node),"sym:weak", "1");
		add_symbols((yyval.node));
	      }
	      Delete((yyvsp[-3].str));
             }
#line 7590 "CParse/parser.c"
    break;

  case 167: /* $@7: %empty  */
#line 4198 "./CParse/parser.y"
                                                                 { 
		    if (currentOuterClass)
		      Setattr(currentOuterClass, "template_parameters", template_parameters);
		    template_parameters = (yyvsp[-1].tparms); 
		    parsing_template_declaration = 1;
		  }
#line 7601 "CParse/parser.c"
    break;

  case 168: /* cpp_template_decl: TEMPLATE LESSTHAN template_parms GREATERTHAN $@7 cpp_template_possible  */
#line 4203 "./CParse/parser.y"
                                          {
			String *tname = 0;
			int     error = 0;

			/* check if we get a namespace node with a class declaration, and retrieve the class */
			Symtab *cscope = Swig_symbol_current();
			Symtab *sti = 0;
			Node *ntop = (yyvsp[0].node);
			Node *ni = ntop;
			SwigType *ntype = ni ? nodeType(ni) : 0;
			while (ni && Strcmp(ntype,"namespace") == 0) {
			  sti = Getattr(ni,"symtab");
			  ni = firstChild(ni);
			  ntype = nodeType(ni);
			}
			if (sti) {
			  Swig_symbol_setscope(sti);
			  Delete(Namespaceprefix);
			  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
			  (yyval.node) = ni;
			} else {
			  (yyval.node) = (yyvsp[0].node);
			}

			if ((yyval.node)) tname = Getattr((yyval.node),"name");
			
			/* Check if the class is a template specialization */
			if (((yyval.node)) && (Strchr(tname,'<')) && (!is_operator(tname))) {
			  /* If a specialization.  Check if defined. */
			  Node *tempn = 0;
			  {
			    String *tbase = SwigType_templateprefix(tname);
			    tempn = Swig_symbol_clookup_local(tbase,0);
			    if (!tempn || (Strcmp(nodeType(tempn),"template") != 0)) {
			      SWIG_WARN_NODE_BEGIN(tempn);
			      Swig_warning(WARN_PARSE_TEMPLATE_SP_UNDEF, Getfile((yyval.node)),Getline((yyval.node)),"Specialization of non-template '%s'.\n", tbase);
			      SWIG_WARN_NODE_END(tempn);
			      tempn = 0;
			      error = 1;
			    }
			    Delete(tbase);
			  }
			  Setattr((yyval.node),"specialization","1");
			  Setattr((yyval.node),"templatetype",nodeType((yyval.node)));
			  set_nodeType((yyval.node),"template");
			  /* Template partial specialization */
			  if (tempn && ((yyvsp[-3].tparms)) && ((yyval.node))) {
			    ParmList *primary_templateparms = Getattr(tempn, "templateparms");
			    String *targs = SwigType_templateargs(tname); /* tname contains name and specialized template parameters, for example: X<(p.T,TT)> */
			    List *tlist = SwigType_parmlist(targs);
			    int specialization_parms_len = Len(tlist);
			    if (!Getattr((yyval.node),"sym:weak")) {
			      Setattr((yyval.node),"sym:typename","1");
			    }
			    Setattr((yyval.node), "primarytemplate", tempn);
			    Setattr((yyval.node), "templateparms", (yyvsp[-3].tparms));
			    Delattr((yyval.node), "specialization");
			    Setattr((yyval.node), "partialspecialization", "1");
			    
			    if (specialization_parms_len > ParmList_len(primary_templateparms)) {
			      Swig_error(Getfile((yyval.node)), Getline((yyval.node)), "Template partial specialization has more arguments than primary template %d %d.\n", specialization_parms_len, ParmList_len(primary_templateparms));
			      
			    } else if (specialization_parms_len < ParmList_numrequired(primary_templateparms)) {
			      Swig_error(Getfile((yyval.node)), Getline((yyval.node)), "Template partial specialization has fewer arguments than primary template %d %d.\n", specialization_parms_len, ParmList_len(primary_templateparms));
			    } else {
			      /* Create a specialized name with template parameters replaced with $ variables, such as, X<(T1,p.T2) => X<($1,p.$2)> */
			      Parm *p = (yyvsp[-3].tparms);
			      String *fname = NewString(tname);
			      String *ffname = 0;
			      ParmList *partialparms = 0;

			      char   tmp[32];
			      int i = 0;
			      while (p) {
				String *name = Getattr(p,"name");
				++i;
				if (!name) {
				  p = nextSibling(p);
				  continue;
				}
				sprintf(tmp, "$%d", i);
				Replaceid(fname, name, tmp);
				p = nextSibling(p);
			      }
			      /* Patch argument names with typedef */
			      {
				Iterator tt;
				Parm *parm_current = 0;
				List *tparms = SwigType_parmlist(fname);
				ffname = SwigType_templateprefix(fname);
				Append(ffname,"<(");
				for (tt = First(tparms); tt.item; ) {
				  SwigType *rtt = Swig_symbol_typedef_reduce(tt.item,0);
				  SwigType *ttr = Swig_symbol_type_qualify(rtt,0);

				  Parm *newp = NewParmWithoutFileLineInfo(ttr, 0);
				  if (partialparms)
				    set_nextSibling(parm_current, newp);
				  else
				    partialparms = newp;
				  parm_current = newp;

				  Append(ffname,ttr);
				  tt = Next(tt);
				  if (tt.item) Putc(',',ffname);
				  Delete(rtt);
				  Delete(ttr);
				}
				Delete(tparms);
				Append(ffname,")>");
			      }
			      {
				/* Replace each primary template parameter's name and value with $ variables, such as, class Y,class T=Y => class $1,class $2=$1 */
				ParmList *primary_templateparms_copy = CopyParmList(primary_templateparms);
				p = primary_templateparms_copy;
				i = 0;
				while (p) {
				  String *name = Getattr(p, "name");
				  Parm *pp = nextSibling(p);
				  ++i;
				  sprintf(tmp, "$%d", i);
				  while (pp) {
				    Replaceid(Getattr(pp, "value"), name, tmp);
				    pp = nextSibling(pp);
				  }
				  Setattr(p, "name", NewString(tmp));
				  p = nextSibling(p);
				}
				/* Modify partialparms by adding in missing default values ($ variables) from primary template parameters */
				partialparms = Swig_cparse_template_partialargs_expand(partialparms, tempn, primary_templateparms_copy);
				Delete(primary_templateparms_copy);
			      }
			      {
				Node *new_partial = NewHash();
				String *partials = Getattr(tempn,"partials");
				if (!partials) {
				  partials = NewList();
				  Setattr(tempn,"partials",partials);
				  Delete(partials);
				}
				/*			      Printf(stdout,"partial: fname = '%s', '%s'\n", fname, Swig_symbol_typedef_reduce(fname,0)); */
				Setattr(new_partial, "partialparms", partialparms);
				Setattr(new_partial, "templcsymname", ffname);
				Append(partials, new_partial);
			      }
			      Setattr((yyval.node),"partialargs",ffname);
			      Swig_symbol_cadd(ffname,(yyval.node));
			    }
			    Delete(tlist);
			    Delete(targs);
			  } else {
			    /* An explicit template specialization */
			    /* add default args from primary (unspecialized) template */
			    String *ty = Swig_symbol_template_deftype(tname,0);
			    String *fname = Swig_symbol_type_qualify(ty,0);
			    Swig_symbol_cadd(fname,(yyval.node));
			    Delete(ty);
			    Delete(fname);
			  }
			} else if ((yyval.node)) {
			  Setattr((yyval.node), "templatetype", nodeType((yyval.node)));
			  set_nodeType((yyval.node),"template");
			  Setattr((yyval.node),"templateparms", (yyvsp[-3].tparms));
			  if (!Getattr((yyval.node),"sym:weak")) {
			    Setattr((yyval.node),"sym:typename","1");
			  }
			  add_symbols((yyval.node));
			  default_arguments((yyval.node));
			  /* We also place a fully parameterized version in the symbol table */
			  {
			    Parm *p;
			    String *fname = NewStringf("%s<(", Getattr((yyval.node),"name"));
			    p = (yyvsp[-3].tparms);
			    while (p) {
			      String *n = Getattr(p,"name");
			      if (!n) n = Getattr(p,"type");
			      Append(fname,n);
			      p = nextSibling(p);
			      if (p) Putc(',',fname);
			    }
			    Append(fname,")>");
			    Swig_symbol_cadd(fname,(yyval.node));
			  }
			}
			(yyval.node) = ntop;
			Swig_symbol_setscope(cscope);
			Delete(Namespaceprefix);
			Namespaceprefix = Swig_symbol_qualifiedscopename(0);
			if (error || (nscope_inner && Strcmp(nodeType(nscope_inner), "class") == 0)) {
			  (yyval.node) = 0;
			}
			if (currentOuterClass)
			  template_parameters = Getattr(currentOuterClass, "template_parameters");
			else
			  template_parameters = 0;
			parsing_template_declaration = 0;
                }
#line 7803 "CParse/parser.c"
    break;

  case 169: /* cpp_template_decl: TEMPLATE cpptype idcolon  */
#line 4402 "./CParse/parser.y"
                                           {
		  Swig_warning(WARN_PARSE_EXPLICIT_TEMPLATE, cparse_file, cparse_line, "Explicit template instantiation ignored.\n");
                  (yyval.node) = 0; 
		}
#line 7812 "CParse/parser.c"
    break;

  case 170: /* cpp_template_decl: TEMPLATE cpp_alternate_rettype idcolon LPAREN parms RPAREN  */
#line 4408 "./CParse/parser.y"
                                                                             {
			Swig_warning(WARN_PARSE_EXPLICIT_TEMPLATE, cparse_file, cparse_line, "Explicit template instantiation ignored.\n");
                  (yyval.node) = 0; 
		}
#line 7821 "CParse/parser.c"
    break;

  case 171: /* cpp_template_decl: EXTERN TEMPLATE cpptype idcolon  */
#line 4414 "./CParse/parser.y"
                                                  {
		  Swig_warning(WARN_PARSE_EXTERN_TEMPLATE, cparse_file, cparse_line, "Extern template ignored.\n");
                  (yyval.node) = 0; 
                }
#line 7830 "CParse/parser.c"
    break;

  case 172: /* cpp_template_decl: EXTERN TEMPLATE cpp_alternate_rettype idcolon LPAREN parms RPAREN  */
#line 4420 "./CParse/parser.y"
                                                                                    {
			Swig_warning(WARN_PARSE_EXTERN_TEMPLATE, cparse_file, cparse_line, "Extern template ignored.\n");
                  (yyval.node) = 0; 
		}
#line 7839 "CParse/parser.c"
    break;

  case 176: /* cpp_template_possible: cpp_template_decl  */
#line 4429 "./CParse/parser.y"
                                    {
		  (yyval.node) = 0;
                }
#line 7847 "CParse/parser.c"
    break;

  case 179: /* template_parms: templateparameter templateparameterstail  */
#line 4436 "./CParse/parser.y"
                                                          {
                      set_nextSibling((yyvsp[-1].p),(yyvsp[0].pl));
                      (yyval.tparms) = (yyvsp[-1].p);
                   }
#line 7856 "CParse/parser.c"
    break;

  case 180: /* template_parms: %empty  */
#line 4440 "./CParse/parser.y"
                            { (yyval.tparms) = 0; }
#line 7862 "CParse/parser.c"
    break;

  case 181: /* templateparameter: templcpptype def_args  */
#line 4443 "./CParse/parser.y"
                                          {
		    (yyval.p) = NewParmWithoutFileLineInfo((yyvsp[-1].type), 0);
		    Setfile((yyval.p), cparse_file);
		    Setline((yyval.p), cparse_line);
		    Setattr((yyval.p), "value", (yyvsp[0].dtype).rawval ? (yyvsp[0].dtype).rawval : (yyvsp[0].dtype).val);
		  }
#line 7873 "CParse/parser.c"
    break;

  case 182: /* templateparameter: TEMPLATE LESSTHAN template_parms GREATERTHAN cpptype idcolon def_args  */
#line 4449 "./CParse/parser.y"
                                                                                          {
		    (yyval.p) = NewParmWithoutFileLineInfo(NewStringf("template< %s > %s %s", ParmList_str_defaultargs((yyvsp[-4].tparms)), (yyvsp[-2].type), (yyvsp[-1].str)), (yyvsp[-1].str));
		    Setfile((yyval.p), cparse_file);
		    Setline((yyval.p), cparse_line);
		    if ((yyvsp[0].dtype).val) {
		      Setattr((yyval.p), "value", (yyvsp[0].dtype).val);
		    }
		  }
#line 7886 "CParse/parser.c"
    break;

  case 183: /* templateparameter: TEMPLATE LESSTHAN template_parms GREATERTHAN cpptype def_args  */
#line 4457 "./CParse/parser.y"
                                                                                  {
		    (yyval.p) = NewParmWithoutFileLineInfo(NewStringf("template< %s > %s", ParmList_str_defaultargs((yyvsp[-3].tparms)), (yyvsp[-1].type)), 0);
		    Setfile((yyval.p), cparse_file);
		    Setline((yyval.p), cparse_line);
		    if ((yyvsp[0].dtype).val) {
		      Setattr((yyval.p), "value", (yyvsp[0].dtype).val);
		    }
		  }
#line 7899 "CParse/parser.c"
    break;

  case 184: /* templateparameter: parm  */
#line 4465 "./CParse/parser.y"
                         {
		    Parm *p = (yyvsp[0].p);
		    String *name = Getattr(p, "name");
		    (yyval.p) = (yyvsp[0].p);

		    /* Correct the 'type name' parameter string, split into the appropriate "name" and "type" attributes */
		    if (!name) {
		      String *type = Getattr(p, "type");
		      if ((Strncmp(type, "class ", 6) == 0) || (Strncmp(type, "typename ", 9) == 0)) {
			/* A 'class T' parameter */
			const char *t = Strchr(type, ' ');
			Setattr(p, "name", t + 1);
			Setattr(p, "type", NewStringWithSize(type, (int)(t - Char(type))));
		      } else if ((Strncmp(type, "v.class ", 8) == 0) || (Strncmp(type, "v.typename ", 11) == 0)) {
			/* Variadic template args */
			const char *t = Strchr(type, ' ');
			Setattr(p, "name", t + 1);
			Setattr(p, "type", NewStringWithSize(type, (int)(t - Char(type))));
		      }
		    }
                  }
#line 7925 "CParse/parser.c"
    break;

  case 185: /* templateparameterstail: COMMA templateparameter templateparameterstail  */
#line 4488 "./CParse/parser.y"
                                                                        {
                         set_nextSibling((yyvsp[-1].p),(yyvsp[0].pl));
                         (yyval.pl) = (yyvsp[-1].p);
                       }
#line 7934 "CParse/parser.c"
    break;

  case 186: /* templateparameterstail: %empty  */
#line 4492 "./CParse/parser.y"
                                { (yyval.pl) = 0; }
#line 7940 "CParse/parser.c"
    break;

  case 187: /* cpp_using_decl: USING idcolon SEMI  */
#line 4497 "./CParse/parser.y"
                                    {
                  String *uname = Swig_symbol_type_qualify((yyvsp[-1].str),0);
                  /* Possible TODO: In testcase using_member_multiple_inherit class Susing3, uname is "Susing1::usingmethod" instead of "Susing2::usingmethod" */
		  String *name = Swig_scopename_last((yyvsp[-1].str));
                  (yyval.node) = new_node("using");
		  Setattr((yyval.node),"uname",uname);
		  Setattr((yyval.node),"name", name);
		  Delete(uname);
		  Delete(name);
		  add_symbols((yyval.node));
             }
#line 7956 "CParse/parser.c"
    break;

  case 188: /* cpp_using_decl: USING TYPENAME idcolon SEMI  */
#line 4508 "./CParse/parser.y"
                                           {
		  String *uname = Swig_symbol_type_qualify((yyvsp[-1].str),0);
		  String *name = Swig_scopename_last((yyvsp[-1].str));
		  (yyval.node) = new_node("using");
		  Setattr((yyval.node),"uname",uname);
		  Setattr((yyval.node),"name", name);
		  Delete(uname);
		  Delete(name);
		  add_symbols((yyval.node));
	     }
#line 7971 "CParse/parser.c"
    break;

  case 189: /* cpp_using_decl: USING NAMESPACE idcolon SEMI  */
#line 4518 "./CParse/parser.y"
                                            {
	       Node *n = Swig_symbol_clookup((yyvsp[-1].str),0);
	       if (!n) {
		 Swig_error(cparse_file, cparse_line, "Nothing known about namespace '%s'\n", SwigType_namestr((yyvsp[-1].str)));
		 (yyval.node) = 0;
	       } else {

		 while (Strcmp(nodeType(n),"using") == 0) {
		   n = Getattr(n,"node");
		 }
		 if (n) {
		   if (Strcmp(nodeType(n),"namespace") == 0) {
		     Symtab *current = Swig_symbol_current();
		     Symtab *symtab = Getattr(n,"symtab");
		     (yyval.node) = new_node("using");
		     Setattr((yyval.node),"node",n);
		     Setattr((yyval.node),"namespace", (yyvsp[-1].str));
		     if (current != symtab) {
		       Swig_symbol_inherit(symtab);
		     }
		   } else {
		     Swig_error(cparse_file, cparse_line, "'%s' is not a namespace.\n", SwigType_namestr((yyvsp[-1].str)));
		     (yyval.node) = 0;
		   }
		 } else {
		   (yyval.node) = 0;
		 }
	       }
             }
#line 8005 "CParse/parser.c"
    break;

  case 190: /* @8: %empty  */
#line 4549 "./CParse/parser.y"
                                                    {
                Hash *h;
		Node *parent_ns = 0;
		List *scopes = Swig_scopename_tolist((yyvsp[-1].str));
		int ilen = Len(scopes);
		int i;

/*
Printf(stdout, "==== Namespace %s creation...\n", $2);
*/
		(yyval.node) = 0;
		for (i = 0; i < ilen; i++) {
		  Node *ns = new_node("namespace");
		  Symtab *current_symtab = Swig_symbol_current();
		  String *scopename = Getitem(scopes, i);
		  Setattr(ns, "name", scopename);
		  (yyval.node) = ns;
		  if (parent_ns)
		    appendChild(parent_ns, ns);
		  parent_ns = ns;
		  h = Swig_symbol_clookup(scopename, 0);
		  if (h && (current_symtab == Getattr(h, "sym:symtab")) && (Strcmp(nodeType(h), "namespace") == 0)) {
/*
Printf(stdout, "  Scope %s [found C++17 style]\n", scopename);
*/
		    if (Getattr(h, "alias")) {
		      h = Getattr(h, "namespace");
		      Swig_warning(WARN_PARSE_NAMESPACE_ALIAS, cparse_file, cparse_line, "Namespace alias '%s' not allowed here. Assuming '%s'\n",
				   scopename, Getattr(h, "name"));
		      scopename = Getattr(h, "name");
		    }
		    Swig_symbol_setscope(Getattr(h, "symtab"));
		  } else {
/*
Printf(stdout, "  Scope %s [creating single scope C++17 style]\n", scopename);
*/
		    h = Swig_symbol_newscope();
		    Swig_symbol_setscopename(scopename);
		  }
		  Delete(Namespaceprefix);
		  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		}
		Delete(scopes);
             }
#line 8054 "CParse/parser.c"
    break;

  case 191: /* cpp_namespace_decl: NAMESPACE idcolon LBRACE @8 interface RBRACE  */
#line 4592 "./CParse/parser.y"
                                {
		Node *n = (yyvsp[-2].node);
		Node *top_ns = 0;
		do {
		  Setattr(n, "symtab", Swig_symbol_popscope());
		  Delete(Namespaceprefix);
		  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		  add_symbols(n);
		  top_ns = n;
		  n = parentNode(n);
		} while(n);
		appendChild((yyvsp[-2].node), firstChild((yyvsp[-1].node)));
		Delete((yyvsp[-1].node));
		(yyval.node) = top_ns;
             }
#line 8074 "CParse/parser.c"
    break;

  case 192: /* @9: %empty  */
#line 4607 "./CParse/parser.y"
                                      {
	       Hash *h;
	       (yyval.node) = Swig_symbol_current();
	       h = Swig_symbol_clookup("    ",0);
	       if (h && (Strcmp(nodeType(h),"namespace") == 0)) {
		 Swig_symbol_setscope(Getattr(h,"symtab"));
	       } else {
		 Swig_symbol_newscope();
		 /* we don't use "__unnamed__", but a long 'empty' name */
		 Swig_symbol_setscopename("    ");
	       }
	       Namespaceprefix = 0;
             }
#line 8092 "CParse/parser.c"
    break;

  case 193: /* cpp_namespace_decl: NAMESPACE LBRACE @9 interface RBRACE  */
#line 4619 "./CParse/parser.y"
                                {
	       (yyval.node) = (yyvsp[-1].node);
	       set_nodeType((yyval.node),"namespace");
	       Setattr((yyval.node),"unnamed","1");
	       Setattr((yyval.node),"symtab", Swig_symbol_popscope());
	       Swig_symbol_setscope((yyvsp[-2].node));
	       Delete(Namespaceprefix);
	       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	       add_symbols((yyval.node));
             }
#line 8107 "CParse/parser.c"
    break;

  case 194: /* cpp_namespace_decl: NAMESPACE identifier EQUAL idcolon SEMI  */
#line 4629 "./CParse/parser.y"
                                                       {
	       /* Namespace alias */
	       Node *n;
	       (yyval.node) = new_node("namespace");
	       Setattr((yyval.node),"name",(yyvsp[-3].id));
	       Setattr((yyval.node),"alias",(yyvsp[-1].str));
	       n = Swig_symbol_clookup((yyvsp[-1].str),0);
	       if (!n) {
		 Swig_error(cparse_file, cparse_line, "Unknown namespace '%s'\n", SwigType_namestr((yyvsp[-1].str)));
		 (yyval.node) = 0;
	       } else {
		 if (Strcmp(nodeType(n),"namespace") != 0) {
		   Swig_error(cparse_file, cparse_line, "'%s' is not a namespace\n", SwigType_namestr((yyvsp[-1].str)));
		   (yyval.node) = 0;
		 } else {
		   while (Getattr(n,"alias")) {
		     n = Getattr(n,"namespace");
		   }
		   Setattr((yyval.node),"namespace",n);
		   add_symbols((yyval.node));
		   /* Set up a scope alias */
		   Swig_symbol_alias((yyvsp[-3].id),Getattr(n,"symtab"));
		 }
	       }
             }
#line 8137 "CParse/parser.c"
    break;

  case 195: /* cpp_members: cpp_member cpp_members  */
#line 4656 "./CParse/parser.y"
                                      {
                   (yyval.node) = (yyvsp[-1].node);
                   /* Insert cpp_member (including any siblings) to the front of the cpp_members linked list */
		   if ((yyval.node)) {
		     Node *p = (yyval.node);
		     Node *pp =0;
		     while (p) {
		       pp = p;
		       p = nextSibling(p);
		     }
		     set_nextSibling(pp,(yyvsp[0].node));
		     if ((yyvsp[0].node))
		       set_previousSibling((yyvsp[0].node), pp);
		   } else {
		     (yyval.node) = (yyvsp[0].node);
		   }
	     }
#line 8159 "CParse/parser.c"
    break;

  case 197: /* $@10: %empty  */
#line 4674 "./CParse/parser.y"
                             { 
	       extendmode = 1;
	       if (cplus_mode != CPLUS_PUBLIC) {
		 Swig_error(cparse_file,cparse_line,"%%extend can only be used in a public section\n");
	       }
             }
#line 8170 "CParse/parser.c"
    break;

  case 198: /* $@11: %empty  */
#line 4679 "./CParse/parser.y"
                                  {
	       extendmode = 0;
	     }
#line 8178 "CParse/parser.c"
    break;

  case 199: /* cpp_members: EXTEND LBRACE $@10 cpp_members RBRACE $@11 cpp_members  */
#line 4681 "./CParse/parser.y"
                           {
	       (yyval.node) = new_node("extend");
	       mark_nodes_as_extend((yyvsp[-3].node));
	       appendChild((yyval.node),(yyvsp[-3].node));
	       set_nextSibling((yyval.node),(yyvsp[0].node));
	     }
#line 8189 "CParse/parser.c"
    break;

  case 201: /* cpp_members: %empty  */
#line 4688 "./CParse/parser.y"
                      { (yyval.node) = 0;}
#line 8195 "CParse/parser.c"
    break;

  case 202: /* cpp_members: error  */
#line 4689 "./CParse/parser.y"
                     {
	       Swig_error(cparse_file,cparse_line,"Syntax error in input(3).\n");
	       Exit(EXIT_FAILURE);
	     }
#line 8204 "CParse/parser.c"
    break;

  case 204: /* cpp_member_no_dox: cpp_constructor_decl  */
#line 4702 "./CParse/parser.y"
                                    { 
                 (yyval.node) = (yyvsp[0].node); 
		 if (extendmode && current_class) {
		   String *symname;
		   symname= make_name((yyval.node),Getattr((yyval.node),"name"), Getattr((yyval.node),"decl"));
		   if (Strcmp(symname,Getattr((yyval.node),"name")) == 0) {
		     /* No renaming operation.  Set name to class name */
		     Delete(yyrename);
		     yyrename = NewString(Getattr(current_class,"sym:name"));
		   } else {
		     Delete(yyrename);
		     yyrename = symname;
		   }
		 }
		 add_symbols((yyval.node));
                 default_arguments((yyval.node));
             }
#line 8226 "CParse/parser.c"
    break;

  case 211: /* cpp_member_no_dox: storage_class idcolon SEMI  */
#line 4725 "./CParse/parser.y"
                                          { (yyval.node) = 0; Delete((yyvsp[-2].str)); }
#line 8232 "CParse/parser.c"
    break;

  case 217: /* cpp_member_no_dox: anonymous_bitfield  */
#line 4731 "./CParse/parser.y"
                                  { (yyval.node) = 0; }
#line 8238 "CParse/parser.c"
    break;

  case 220: /* cpp_member_no_dox: SEMI  */
#line 4734 "./CParse/parser.y"
                    { (yyval.node) = 0; }
#line 8244 "CParse/parser.c"
    break;

  case 222: /* cpp_member: DOXYGENSTRING cpp_member_no_dox  */
#line 4737 "./CParse/parser.y"
                                               {
	         (yyval.node) = (yyvsp[0].node);
		 set_comment((yyvsp[0].node), (yyvsp[-1].str));
	     }
#line 8253 "CParse/parser.c"
    break;

  case 223: /* cpp_member: cpp_member_no_dox DOXYGENPOSTSTRING  */
#line 4741 "./CParse/parser.y"
                                                   {
	         (yyval.node) = (yyvsp[-1].node);
		 set_comment((yyvsp[-1].node), (yyvsp[0].str));
	     }
#line 8262 "CParse/parser.c"
    break;

  case 224: /* cpp_constructor_decl: storage_class type LPAREN parms RPAREN ctor_end  */
#line 4753 "./CParse/parser.y"
                                                                       {
	      /* Cannot be a constructor declaration/definition if parsed as a friend destructor/constructor
	         or a badly declared friend function without return type */
	      int isfriend = Strstr((yyvsp[-5].str), "friend") != NULL;
	      if (!isfriend && (inclass || extendmode)) {
	        String *name = SwigType_templateprefix((yyvsp[-4].type)); /* A constructor can optionally be declared with template parameters before C++20, strip these off */
		SwigType *decl = NewStringEmpty();
		(yyval.node) = new_node("constructor");
		Setattr((yyval.node),"storage",(yyvsp[-5].str));
		Setattr((yyval.node), "name", name);
		Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		SwigType_add_function(decl,(yyvsp[-2].pl));
		Setattr((yyval.node),"decl",decl);
		Setattr((yyval.node),"throws",(yyvsp[0].decl).throws);
		Setattr((yyval.node),"throw",(yyvsp[0].decl).throwf);
		Setattr((yyval.node),"noexcept",(yyvsp[0].decl).nexcept);
		Setattr((yyval.node),"final",(yyvsp[0].decl).final);
		if (Len(scanner_ccode)) {
		  String *code = Copy(scanner_ccode);
		  Setattr((yyval.node),"code",code);
		  Delete(code);
		}
		SetFlag((yyval.node),"feature:new");
		if ((yyvsp[0].decl).defarg)
		  Setattr((yyval.node),"value",(yyvsp[0].decl).defarg);
	      } else {
		(yyval.node) = 0;
		Delete((yyvsp[-5].str));
              }
              }
#line 8297 "CParse/parser.c"
    break;

  case 225: /* cpp_destructor_decl: storage_class NOT idtemplate LPAREN parms RPAREN cpp_vend  */
#line 4787 "./CParse/parser.y"
                                                                                {
	       String *name = SwigType_templateprefix((yyvsp[-4].str)); /* A destructor can optionally be declared with template parameters before C++20, strip these off */
	       Insert(name, 0, "~");
	       (yyval.node) = new_node("destructor");
	       Setattr((yyval.node), "storage", (yyvsp[-6].str));
	       Setattr((yyval.node), "name", name);
	       Delete(name);
	       if (Len(scanner_ccode)) {
		 String *code = Copy(scanner_ccode);
		 Setattr((yyval.node), "code", code);
		 Delete(code);
	       }
	       {
		 String *decl = NewStringEmpty();
		 SwigType_add_function(decl, (yyvsp[-2].pl));
		 Setattr((yyval.node), "decl", decl);
		 Delete(decl);
	       }
	       Setattr((yyval.node), "throws", (yyvsp[0].dtype).throws);
	       Setattr((yyval.node), "throw", (yyvsp[0].dtype).throwf);
	       Setattr((yyval.node), "noexcept", (yyvsp[0].dtype).nexcept);
	       Setattr((yyval.node), "final", (yyvsp[0].dtype).final);
	       if ((yyvsp[0].dtype).val) {
		 if (Equal((yyvsp[0].dtype).val, "0")) {
		   if (!Strstr((yyvsp[-6].str), "virtual"))
		     Swig_error(cparse_file, cparse_line, "Destructor %s uses a pure specifier but is not virtual.\n", Swig_name_decl((yyval.node)));
		 } else if (!(Equal((yyvsp[0].dtype).val, "delete") || Equal((yyvsp[0].dtype).val, "default"))) {
		   Swig_error(cparse_file, cparse_line, "Destructor %s has an invalid pure specifier, only = 0 is allowed.\n", Swig_name_decl((yyval.node)));
		 }
		 Setattr((yyval.node), "value", (yyvsp[0].dtype).val);
	       }
	       /* TODO: check all storage decl-specifiers are valid */
	       if ((yyvsp[0].dtype).qualifier)
		 Swig_error(cparse_file, cparse_line, "Destructor %s %s cannot have a qualifier.\n", Swig_name_decl((yyval.node)), SwigType_str((yyvsp[0].dtype).qualifier, 0));
	       add_symbols((yyval.node));
	      }
#line 8338 "CParse/parser.c"
    break;

  case 226: /* cpp_conversion_operator: storage_class CONVERSIONOPERATOR type pointer LPAREN parms RPAREN cpp_vend  */
#line 4827 "./CParse/parser.y"
                                                                                                     {
                 (yyval.node) = new_node("cdecl");
                 Setattr((yyval.node),"type",(yyvsp[-5].type));
		 Setattr((yyval.node),"name",(yyvsp[-6].str));
		 Setattr((yyval.node),"storage",(yyvsp[-7].str));

		 SwigType_add_function((yyvsp[-4].type),(yyvsp[-2].pl));
		 if ((yyvsp[0].dtype).qualifier) {
		   SwigType_push((yyvsp[-4].type),(yyvsp[0].dtype).qualifier);
		 }
		 if ((yyvsp[0].dtype).val) {
		   Setattr((yyval.node),"value",(yyvsp[0].dtype).val);
		 }
		 Setattr((yyval.node),"refqualifier",(yyvsp[0].dtype).refqualifier);
		 Setattr((yyval.node),"decl",(yyvsp[-4].type));
		 Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		 Setattr((yyval.node),"conversion_operator","1");
		 add_symbols((yyval.node));
              }
#line 8362 "CParse/parser.c"
    break;

  case 227: /* cpp_conversion_operator: storage_class CONVERSIONOPERATOR type AND LPAREN parms RPAREN cpp_vend  */
#line 4846 "./CParse/parser.y"
                                                                                        {
		 SwigType *decl;
                 (yyval.node) = new_node("cdecl");
                 Setattr((yyval.node),"type",(yyvsp[-5].type));
		 Setattr((yyval.node),"name",(yyvsp[-6].str));
		 Setattr((yyval.node),"storage",(yyvsp[-7].str));
		 decl = NewStringEmpty();
		 SwigType_add_reference(decl);
		 SwigType_add_function(decl,(yyvsp[-2].pl));
		 if ((yyvsp[0].dtype).qualifier) {
		   SwigType_push(decl,(yyvsp[0].dtype).qualifier);
		 }
		 if ((yyvsp[0].dtype).val) {
		   Setattr((yyval.node),"value",(yyvsp[0].dtype).val);
		 }
		 Setattr((yyval.node),"refqualifier",(yyvsp[0].dtype).refqualifier);
		 Setattr((yyval.node),"decl",decl);
		 Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		 Setattr((yyval.node),"conversion_operator","1");
		 add_symbols((yyval.node));
	       }
#line 8388 "CParse/parser.c"
    break;

  case 228: /* cpp_conversion_operator: storage_class CONVERSIONOPERATOR type LAND LPAREN parms RPAREN cpp_vend  */
#line 4867 "./CParse/parser.y"
                                                                                         {
		 SwigType *decl;
                 (yyval.node) = new_node("cdecl");
                 Setattr((yyval.node),"type",(yyvsp[-5].type));
		 Setattr((yyval.node),"name",(yyvsp[-6].str));
		 Setattr((yyval.node),"storage",(yyvsp[-7].str));
		 decl = NewStringEmpty();
		 SwigType_add_rvalue_reference(decl);
		 SwigType_add_function(decl,(yyvsp[-2].pl));
		 if ((yyvsp[0].dtype).qualifier) {
		   SwigType_push(decl,(yyvsp[0].dtype).qualifier);
		 }
		 if ((yyvsp[0].dtype).val) {
		   Setattr((yyval.node),"value",(yyvsp[0].dtype).val);
		 }
		 Setattr((yyval.node),"refqualifier",(yyvsp[0].dtype).refqualifier);
		 Setattr((yyval.node),"decl",decl);
		 Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		 Setattr((yyval.node),"conversion_operator","1");
		 add_symbols((yyval.node));
	       }
#line 8414 "CParse/parser.c"
    break;

  case 229: /* cpp_conversion_operator: storage_class CONVERSIONOPERATOR type pointer AND LPAREN parms RPAREN cpp_vend  */
#line 4889 "./CParse/parser.y"
                                                                                                {
		 SwigType *decl;
                 (yyval.node) = new_node("cdecl");
                 Setattr((yyval.node),"type",(yyvsp[-6].type));
		 Setattr((yyval.node),"name",(yyvsp[-7].str));
		 Setattr((yyval.node),"storage",(yyvsp[-8].str));
		 decl = NewStringEmpty();
		 SwigType_add_pointer(decl);
		 SwigType_add_reference(decl);
		 SwigType_add_function(decl,(yyvsp[-2].pl));
		 if ((yyvsp[0].dtype).qualifier) {
		   SwigType_push(decl,(yyvsp[0].dtype).qualifier);
		 }
		 if ((yyvsp[0].dtype).val) {
		   Setattr((yyval.node),"value",(yyvsp[0].dtype).val);
		 }
		 Setattr((yyval.node),"refqualifier",(yyvsp[0].dtype).refqualifier);
		 Setattr((yyval.node),"decl",decl);
		 Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		 Setattr((yyval.node),"conversion_operator","1");
		 add_symbols((yyval.node));
	       }
#line 8441 "CParse/parser.c"
    break;

  case 230: /* cpp_conversion_operator: storage_class CONVERSIONOPERATOR type LPAREN parms RPAREN cpp_vend  */
#line 4912 "./CParse/parser.y"
                                                                                   {
		String *t = NewStringEmpty();
		(yyval.node) = new_node("cdecl");
		Setattr((yyval.node),"type",(yyvsp[-4].type));
		Setattr((yyval.node),"name",(yyvsp[-5].str));
		 Setattr((yyval.node),"storage",(yyvsp[-6].str));
		SwigType_add_function(t,(yyvsp[-2].pl));
		if ((yyvsp[0].dtype).qualifier) {
		  SwigType_push(t,(yyvsp[0].dtype).qualifier);
		}
		if ((yyvsp[0].dtype).val) {
		  Setattr((yyval.node),"value",(yyvsp[0].dtype).val);
		}
		Setattr((yyval.node),"refqualifier",(yyvsp[0].dtype).refqualifier);
		Setattr((yyval.node),"decl",t);
		Setattr((yyval.node),"parms",(yyvsp[-2].pl));
		Setattr((yyval.node),"conversion_operator","1");
		add_symbols((yyval.node));
              }
#line 8465 "CParse/parser.c"
    break;

  case 231: /* cpp_catch_decl: CATCH LPAREN parms RPAREN LBRACE  */
#line 4935 "./CParse/parser.y"
                                                  {
                 if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
                 (yyval.node) = 0;
               }
#line 8474 "CParse/parser.c"
    break;

  case 232: /* cpp_static_assert: STATIC_ASSERT LPAREN  */
#line 4943 "./CParse/parser.y"
                                         {
                if (skip_balanced('(',')') < 0) Exit(EXIT_FAILURE);
                (yyval.node) = 0;
              }
#line 8483 "CParse/parser.c"
    break;

  case 233: /* cpp_protection_decl: PUBLIC COLON  */
#line 4950 "./CParse/parser.y"
                                   { 
                (yyval.node) = new_node("access");
		Setattr((yyval.node),"kind","public");
                cplus_mode = CPLUS_PUBLIC;
              }
#line 8493 "CParse/parser.c"
    break;

  case 234: /* cpp_protection_decl: PRIVATE COLON  */
#line 4957 "./CParse/parser.y"
                              { 
                (yyval.node) = new_node("access");
                Setattr((yyval.node),"kind","private");
		cplus_mode = CPLUS_PRIVATE;
	      }
#line 8503 "CParse/parser.c"
    break;

  case 235: /* cpp_protection_decl: PROTECTED COLON  */
#line 4965 "./CParse/parser.y"
                                { 
		(yyval.node) = new_node("access");
		Setattr((yyval.node),"kind","protected");
		cplus_mode = CPLUS_PROTECTED;
	      }
#line 8513 "CParse/parser.c"
    break;

  case 246: /* cpp_vend: cpp_const SEMI  */
#line 4988 "./CParse/parser.y"
                                { 
                     Clear(scanner_ccode);
                     (yyval.dtype).val = 0;
                     (yyval.dtype).qualifier = (yyvsp[-1].dtype).qualifier;
                     (yyval.dtype).refqualifier = (yyvsp[-1].dtype).refqualifier;
                     (yyval.dtype).bitfield = 0;
                     (yyval.dtype).throws = (yyvsp[-1].dtype).throws;
                     (yyval.dtype).throwf = (yyvsp[-1].dtype).throwf;
                     (yyval.dtype).nexcept = (yyvsp[-1].dtype).nexcept;
                     (yyval.dtype).final = (yyvsp[-1].dtype).final;
                }
#line 8529 "CParse/parser.c"
    break;

  case 247: /* cpp_vend: cpp_const EQUAL definetype SEMI  */
#line 4999 "./CParse/parser.y"
                                                 { 
                     Clear(scanner_ccode);
                     (yyval.dtype).val = (yyvsp[-1].dtype).val;
                     (yyval.dtype).qualifier = (yyvsp[-3].dtype).qualifier;
                     (yyval.dtype).refqualifier = (yyvsp[-3].dtype).refqualifier;
                     (yyval.dtype).bitfield = 0;
                     (yyval.dtype).throws = (yyvsp[-3].dtype).throws; 
                     (yyval.dtype).throwf = (yyvsp[-3].dtype).throwf; 
                     (yyval.dtype).nexcept = (yyvsp[-3].dtype).nexcept;
                     (yyval.dtype).final = (yyvsp[-3].dtype).final;
               }
#line 8545 "CParse/parser.c"
    break;

  case 248: /* cpp_vend: cpp_const LBRACE  */
#line 5010 "./CParse/parser.y"
                                  { 
                     if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
                     (yyval.dtype).val = 0;
                     (yyval.dtype).qualifier = (yyvsp[-1].dtype).qualifier;
                     (yyval.dtype).refqualifier = (yyvsp[-1].dtype).refqualifier;
                     (yyval.dtype).bitfield = 0;
                     (yyval.dtype).throws = (yyvsp[-1].dtype).throws; 
                     (yyval.dtype).throwf = (yyvsp[-1].dtype).throwf; 
                     (yyval.dtype).nexcept = (yyvsp[-1].dtype).nexcept;
                     (yyval.dtype).final = (yyvsp[-1].dtype).final;
               }
#line 8561 "CParse/parser.c"
    break;

  case 249: /* anonymous_bitfield: storage_class anon_bitfield_type COLON expr SEMI  */
#line 5024 "./CParse/parser.y"
                                                                       { Delete((yyvsp[-4].str)); }
#line 8567 "CParse/parser.c"
    break;

  case 254: /* anon_bitfield_type: idcolon  */
#line 5032 "./CParse/parser.y"
                         { (yyval.type) = (yyvsp[0].str); }
#line 8573 "CParse/parser.c"
    break;

  case 255: /* storage_class: storage_class_list  */
#line 5038 "./CParse/parser.y"
                                    {
		 String *r = NewStringEmpty();

		 /* Check for invalid combinations. */
		 if (multiple_bits_set((yyvsp[0].intvalue) & (SWIG_STORAGE_CLASS_EXTERN |
					     SWIG_STORAGE_CLASS_STATIC))) {
		   Swig_error(cparse_file, cparse_line, "Storage class can't be both 'static' and 'extern'");
		 }
		 if (multiple_bits_set((yyvsp[0].intvalue) & (SWIG_STORAGE_CLASS_EXTERNC |
					     SWIG_STORAGE_CLASS_EXTERN |
					     SWIG_STORAGE_CLASS_EXTERNCPP))) {
		   Swig_error(cparse_file, cparse_line, "Declaration can only be one of 'extern', 'extern \"C\"' and 'extern \"C++\"'");
		 }

		 if ((yyvsp[0].intvalue) & SWIG_STORAGE_CLASS_TYPEDEF) {
		   Append(r, "typedef ");
		 } else {
		   if ((yyvsp[0].intvalue) & SWIG_STORAGE_CLASS_EXTERNC)
		     Append(r, "externc ");
		   if ((yyvsp[0].intvalue) & (SWIG_STORAGE_CLASS_EXTERN|SWIG_STORAGE_CLASS_EXTERNCPP))
		     Append(r, "extern ");
		   if ((yyvsp[0].intvalue) & SWIG_STORAGE_CLASS_STATIC)
		     Append(r, "static ");
		 }
		 if ((yyvsp[0].intvalue) & SWIG_STORAGE_CLASS_VIRTUAL)
		   Append(r, "virtual ");
		 if ((yyvsp[0].intvalue) & SWIG_STORAGE_CLASS_FRIEND)
		   Append(r, "friend ");
		 if ((yyvsp[0].intvalue) & SWIG_STORAGE_CLASS_EXPLICIT)
		   Append(r, "explicit ");
		 if ((yyvsp[0].intvalue) & SWIG_STORAGE_CLASS_CONSTEXPR)
		   Append(r, "constexpr ");
		 if ((yyvsp[0].intvalue) & SWIG_STORAGE_CLASS_THREAD_LOCAL)
		   Append(r, "thread_local ");
		 if (Len(r) == 0) {
		   Delete(r);
		   (yyval.str) = 0;
		 } else {
		   Chop(r);
		   (yyval.str) = r;
		 }
	       }
#line 8620 "CParse/parser.c"
    break;

  case 256: /* storage_class: %empty  */
#line 5080 "./CParse/parser.y"
                        { (yyval.str) = 0; }
#line 8626 "CParse/parser.c"
    break;

  case 258: /* storage_class_list: storage_class_list storage_class_raw  */
#line 5084 "./CParse/parser.y"
                                                      {
		  if ((yyvsp[-1].intvalue) & (yyvsp[0].intvalue)) {
		    Swig_error(cparse_file, cparse_line, "Repeated storage class or type specifier '%s'\n", storage_class_string((yyvsp[0].intvalue)));
		  }
		  (yyval.intvalue) = (yyvsp[-1].intvalue) | (yyvsp[0].intvalue);
	       }
#line 8637 "CParse/parser.c"
    break;

  case 259: /* storage_class_raw: EXTERN  */
#line 5092 "./CParse/parser.y"
                            { (yyval.intvalue) = SWIG_STORAGE_CLASS_EXTERN; }
#line 8643 "CParse/parser.c"
    break;

  case 260: /* storage_class_raw: EXTERN string  */
#line 5093 "./CParse/parser.y"
                               {
		   if (Strcmp((yyvsp[0].str),"C") == 0) {
		     (yyval.intvalue) = SWIG_STORAGE_CLASS_EXTERNC;
		   } else if (Strcmp((yyvsp[0].str),"C++") == 0) {
		     (yyval.intvalue) = SWIG_STORAGE_CLASS_EXTERNCPP;
		   } else {
		     Swig_warning(WARN_PARSE_UNDEFINED_EXTERN,cparse_file, cparse_line,"Unrecognized extern type \"%s\".\n", (yyvsp[0].str));
		     (yyval.intvalue) = 0;
		   }
	       }
#line 8658 "CParse/parser.c"
    break;

  case 261: /* storage_class_raw: STATIC  */
#line 5103 "./CParse/parser.y"
                        { (yyval.intvalue) = SWIG_STORAGE_CLASS_STATIC; }
#line 8664 "CParse/parser.c"
    break;

  case 262: /* storage_class_raw: TYPEDEF  */
#line 5104 "./CParse/parser.y"
                         { (yyval.intvalue) = SWIG_STORAGE_CLASS_TYPEDEF; }
#line 8670 "CParse/parser.c"
    break;

  case 263: /* storage_class_raw: VIRTUAL  */
#line 5105 "./CParse/parser.y"
                         { (yyval.intvalue) = SWIG_STORAGE_CLASS_VIRTUAL; }
#line 8676 "CParse/parser.c"
    break;

  case 264: /* storage_class_raw: FRIEND  */
#line 5106 "./CParse/parser.y"
                        { (yyval.intvalue) = SWIG_STORAGE_CLASS_FRIEND; }
#line 8682 "CParse/parser.c"
    break;

  case 265: /* storage_class_raw: EXPLICIT  */
#line 5107 "./CParse/parser.y"
                          { (yyval.intvalue) = SWIG_STORAGE_CLASS_EXPLICIT; }
#line 8688 "CParse/parser.c"
    break;

  case 266: /* storage_class_raw: CONSTEXPR  */
#line 5108 "./CParse/parser.y"
                           { (yyval.intvalue) = SWIG_STORAGE_CLASS_CONSTEXPR; }
#line 8694 "CParse/parser.c"
    break;

  case 267: /* storage_class_raw: THREAD_LOCAL  */
#line 5109 "./CParse/parser.y"
                              { (yyval.intvalue) = SWIG_STORAGE_CLASS_THREAD_LOCAL; }
#line 8700 "CParse/parser.c"
    break;

  case 268: /* parms: rawparms  */
#line 5116 "./CParse/parser.y"
                          {
                 Parm *p;
		 (yyval.pl) = (yyvsp[0].pl);
		 p = (yyvsp[0].pl);
                 while (p) {
		   Replace(Getattr(p,"type"),"typename ", "", DOH_REPLACE_ANY);
		   p = nextSibling(p);
                 }
               }
#line 8714 "CParse/parser.c"
    break;

  case 269: /* rawparms: parm  */
#line 5128 "./CParse/parser.y"
                       { (yyval.pl) = (yyvsp[0].p); }
#line 8720 "CParse/parser.c"
    break;

  case 270: /* rawparms: parm DOXYGENPOSTSTRING  */
#line 5129 "./CParse/parser.y"
                                         {
		  set_comment((yyvsp[-1].p), (yyvsp[0].str));
		  (yyval.pl) = (yyvsp[-1].p);
		}
#line 8729 "CParse/parser.c"
    break;

  case 271: /* rawparms: parm DOXYGENSTRING  */
#line 5133 "./CParse/parser.y"
                                     {
		  /* Misplaced doxygen string, attach it to previous parameter, like Doxygen does */
		  set_comment((yyvsp[-1].p), (yyvsp[0].str));
		  (yyval.pl) = (yyvsp[-1].p);
		}
#line 8739 "CParse/parser.c"
    break;

  case 272: /* rawparms: parm COMMA parms  */
#line 5138 "./CParse/parser.y"
                                   {
		  if ((yyvsp[0].pl)) {
		    set_nextSibling((yyvsp[-2].p), (yyvsp[0].pl));
		  }
		  (yyval.pl) = (yyvsp[-2].p);
		}
#line 8750 "CParse/parser.c"
    break;

  case 273: /* rawparms: parm DOXYGENPOSTSTRING COMMA parms  */
#line 5144 "./CParse/parser.y"
                                                     {
		  if ((yyvsp[0].pl)) {
		    set_nextSibling((yyvsp[-3].p), (yyvsp[0].pl));
		  }
		  set_comment((yyvsp[-3].p), (yyvsp[-2].str));
		  (yyval.pl) = (yyvsp[-3].p);
		}
#line 8762 "CParse/parser.c"
    break;

  case 274: /* rawparms: parm COMMA DOXYGENPOSTSTRING parms  */
#line 5151 "./CParse/parser.y"
                                                     {
		  if ((yyvsp[0].pl)) {
		    set_nextSibling((yyvsp[-3].p), (yyvsp[0].pl));
		  }
		  set_comment((yyvsp[-3].p), (yyvsp[-1].str));
		  (yyval.pl) = (yyvsp[-3].p);
		}
#line 8774 "CParse/parser.c"
    break;

  case 275: /* rawparms: %empty  */
#line 5158 "./CParse/parser.y"
                         {
		  (yyval.pl) = 0;
		}
#line 8782 "CParse/parser.c"
    break;

  case 276: /* parm_no_dox: rawtype parameter_declarator  */
#line 5163 "./CParse/parser.y"
                                               {
                   SwigType_push((yyvsp[-1].type),(yyvsp[0].decl).type);
		   (yyval.p) = NewParmWithoutFileLineInfo((yyvsp[-1].type),(yyvsp[0].decl).id);
		   Setfile((yyval.p),cparse_file);
		   Setline((yyval.p),cparse_line);
		   if ((yyvsp[0].decl).defarg) {
		     Setattr((yyval.p),"value",(yyvsp[0].decl).defarg);
		   }
		}
#line 8796 "CParse/parser.c"
    break;

  case 277: /* parm_no_dox: ELLIPSIS  */
#line 5172 "./CParse/parser.y"
                           {
		  SwigType *t = NewString("v(...)");
		  (yyval.p) = NewParmWithoutFileLineInfo(t, 0);
		  Setfile((yyval.p),cparse_file);
		  Setline((yyval.p),cparse_line);
		}
#line 8807 "CParse/parser.c"
    break;

  case 279: /* parm: DOXYGENSTRING parm_no_dox  */
#line 5181 "./CParse/parser.y"
                                            {
		  (yyval.p) = (yyvsp[0].p);
		  set_comment((yyvsp[0].p), (yyvsp[-1].str));
		}
#line 8816 "CParse/parser.c"
    break;

  case 280: /* valparms: rawvalparms  */
#line 5187 "./CParse/parser.y"
                              {
                 Parm *p;
		 (yyval.p) = (yyvsp[0].p);
		 p = (yyvsp[0].p);
                 while (p) {
		   if (Getattr(p,"type")) {
		     Replace(Getattr(p,"type"),"typename ", "", DOH_REPLACE_ANY);
		   }
		   p = nextSibling(p);
                 }
               }
#line 8832 "CParse/parser.c"
    break;

  case 281: /* rawvalparms: valparm valptail  */
#line 5200 "./CParse/parser.y"
                                   {
                  set_nextSibling((yyvsp[-1].p),(yyvsp[0].p));
                  (yyval.p) = (yyvsp[-1].p);
		}
#line 8841 "CParse/parser.c"
    break;

  case 282: /* rawvalparms: %empty  */
#line 5204 "./CParse/parser.y"
                        { (yyval.p) = 0; }
#line 8847 "CParse/parser.c"
    break;

  case 283: /* valptail: COMMA valparm valptail  */
#line 5207 "./CParse/parser.y"
                                        {
                 set_nextSibling((yyvsp[-1].p),(yyvsp[0].p));
		 (yyval.p) = (yyvsp[-1].p);
                }
#line 8856 "CParse/parser.c"
    break;

  case 284: /* valptail: %empty  */
#line 5211 "./CParse/parser.y"
                        { (yyval.p) = 0; }
#line 8862 "CParse/parser.c"
    break;

  case 285: /* valparm: parm  */
#line 5215 "./CParse/parser.y"
                      {
		  (yyval.p) = (yyvsp[0].p);
		  {
		    /* We need to make a possible adjustment for integer parameters. */
		    SwigType *type;
		    Node     *n = 0;

		    while (!n) {
		      type = Getattr((yyvsp[0].p),"type");
		      n = Swig_symbol_clookup(type,0);     /* See if we can find a node that matches the typename */
		      if ((n) && (Strcmp(nodeType(n),"cdecl") == 0)) {
			SwigType *decl = Getattr(n,"decl");
			if (!SwigType_isfunction(decl)) {
			  String *value = Getattr(n,"value");
			  if (value) {
			    String *v = Copy(value);
			    Setattr((yyvsp[0].p),"type",v);
			    Delete(v);
			    n = 0;
			  }
			}
		      } else {
			break;
		      }
		    }
		  }

               }
#line 8895 "CParse/parser.c"
    break;

  case 286: /* valparm: valexpr  */
#line 5243 "./CParse/parser.y"
                         {
                  (yyval.p) = NewParmWithoutFileLineInfo(0,0);
                  Setfile((yyval.p),cparse_file);
		  Setline((yyval.p),cparse_line);
		  Setattr((yyval.p),"value",(yyvsp[0].dtype).val);
               }
#line 8906 "CParse/parser.c"
    break;

  case 287: /* callparms: valexpr callptail  */
#line 5251 "./CParse/parser.y"
                                   {
		 (yyval.dtype) = (yyvsp[-1].dtype);
		 Printf((yyval.dtype).val, "%s", (yyvsp[0].dtype).val);
	       }
#line 8915 "CParse/parser.c"
    break;

  case 288: /* callparms: %empty  */
#line 5255 "./CParse/parser.y"
                        { (yyval.dtype).val = NewStringEmpty(); }
#line 8921 "CParse/parser.c"
    break;

  case 289: /* callptail: COMMA valexpr callptail  */
#line 5258 "./CParse/parser.y"
                                         {
		 (yyval.dtype).val = NewStringf(",%s%s", (yyvsp[-1].dtype).val, (yyvsp[0].dtype).val);
		 (yyval.dtype).type = 0;
	       }
#line 8930 "CParse/parser.c"
    break;

  case 290: /* callptail: %empty  */
#line 5262 "./CParse/parser.y"
                        { (yyval.dtype).val = NewStringEmpty(); }
#line 8936 "CParse/parser.c"
    break;

  case 291: /* def_args: EQUAL definetype  */
#line 5265 "./CParse/parser.y"
                                  { 
                 (yyval.dtype) = (yyvsp[0].dtype);
               }
#line 8944 "CParse/parser.c"
    break;

  case 292: /* def_args: EQUAL definetype LBRACKET expr RBRACKET  */
#line 5268 "./CParse/parser.y"
                                                         { 
		 (yyval.dtype) = (yyvsp[-3].dtype);
		 (yyval.dtype).val = NewStringf("%s[%s]", (yyvsp[-3].dtype).val, (yyvsp[-1].dtype).val);
               }
#line 8953 "CParse/parser.c"
    break;

  case 293: /* def_args: EQUAL LBRACE  */
#line 5272 "./CParse/parser.y"
                              {
		 if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		 (yyval.dtype).val = NewString(scanner_ccode);
		 (yyval.dtype).rawval = 0;
		 (yyval.dtype).type = T_UNKNOWN;
		 (yyval.dtype).unary_arg_type = 0;
		 (yyval.dtype).bitfield = 0;
		 (yyval.dtype).throws = 0;
		 (yyval.dtype).throwf = 0;
		 (yyval.dtype).nexcept = 0;
		 (yyval.dtype).final = 0;
	       }
#line 8970 "CParse/parser.c"
    break;

  case 294: /* def_args: COLON expr  */
#line 5284 "./CParse/parser.y"
                            { 
		 (yyval.dtype).val = 0;
		 (yyval.dtype).rawval = 0;
		 (yyval.dtype).type = 0;
		 (yyval.dtype).bitfield = (yyvsp[0].dtype).val;
		 (yyval.dtype).throws = 0;
		 (yyval.dtype).throwf = 0;
		 (yyval.dtype).nexcept = 0;
		 (yyval.dtype).final = 0;
	       }
#line 8985 "CParse/parser.c"
    break;

  case 295: /* def_args: %empty  */
#line 5294 "./CParse/parser.y"
                        {
                 (yyval.dtype).val = 0;
                 (yyval.dtype).rawval = 0;
                 (yyval.dtype).type = T_UNKNOWN;
		 (yyval.dtype).unary_arg_type = 0;
		 (yyval.dtype).bitfield = 0;
		 (yyval.dtype).throws = 0;
		 (yyval.dtype).throwf = 0;
		 (yyval.dtype).nexcept = 0;
		 (yyval.dtype).final = 0;
               }
#line 9001 "CParse/parser.c"
    break;

  case 296: /* parameter_declarator: declarator def_args  */
#line 5307 "./CParse/parser.y"
                                           {
                 (yyval.decl) = (yyvsp[-1].decl);
		 (yyval.decl).defarg = (yyvsp[0].dtype).rawval ? (yyvsp[0].dtype).rawval : (yyvsp[0].dtype).val;
            }
#line 9010 "CParse/parser.c"
    break;

  case 297: /* parameter_declarator: abstract_declarator def_args  */
#line 5311 "./CParse/parser.y"
                                           {
              (yyval.decl) = (yyvsp[-1].decl);
	      (yyval.decl).defarg = (yyvsp[0].dtype).rawval ? (yyvsp[0].dtype).rawval : (yyvsp[0].dtype).val;
            }
#line 9019 "CParse/parser.c"
    break;

  case 298: /* parameter_declarator: def_args  */
#line 5315 "./CParse/parser.y"
                       {
   	      (yyval.decl).type = 0;
              (yyval.decl).id = 0;
	      (yyval.decl).defarg = (yyvsp[0].dtype).rawval ? (yyvsp[0].dtype).rawval : (yyvsp[0].dtype).val;
            }
#line 9029 "CParse/parser.c"
    break;

  case 299: /* parameter_declarator: direct_declarator LPAREN parms RPAREN cv_ref_qualifier  */
#line 5322 "./CParse/parser.y"
                                                                     {
	      SwigType *t;
	      (yyval.decl) = (yyvsp[-4].decl);
	      t = NewStringEmpty();
	      SwigType_add_function(t,(yyvsp[-2].pl));
	      if ((yyvsp[0].dtype).qualifier)
	        SwigType_push(t, (yyvsp[0].dtype).qualifier);
	      if (!(yyval.decl).have_parms) {
		(yyval.decl).parms = (yyvsp[-2].pl);
		(yyval.decl).have_parms = 1;
	      }
	      if (!(yyval.decl).type) {
		(yyval.decl).type = t;
	      } else {
		SwigType_push(t, (yyval.decl).type);
		Delete((yyval.decl).type);
		(yyval.decl).type = t;
	      }
	      (yyval.decl).defarg = 0;
	    }
#line 9054 "CParse/parser.c"
    break;

  case 300: /* plain_declarator: declarator  */
#line 5344 "./CParse/parser.y"
                              {
                 (yyval.decl) = (yyvsp[0].decl);
		 if (SwigType_isfunction((yyvsp[0].decl).type)) {
		   Delete(SwigType_pop_function((yyvsp[0].decl).type));
		 } else if (SwigType_isarray((yyvsp[0].decl).type)) {
		   SwigType *ta = SwigType_pop_arrays((yyvsp[0].decl).type);
		   if (SwigType_isfunction((yyvsp[0].decl).type)) {
		     Delete(SwigType_pop_function((yyvsp[0].decl).type));
		   } else {
		     (yyval.decl).parms = 0;
		   }
		   SwigType_push((yyvsp[0].decl).type,ta);
		   Delete(ta);
		 } else {
		   (yyval.decl).parms = 0;
		 }
            }
#line 9076 "CParse/parser.c"
    break;

  case 301: /* plain_declarator: abstract_declarator  */
#line 5361 "./CParse/parser.y"
                                  {
              (yyval.decl) = (yyvsp[0].decl);
	      if (SwigType_isfunction((yyvsp[0].decl).type)) {
		Delete(SwigType_pop_function((yyvsp[0].decl).type));
	      } else if (SwigType_isarray((yyvsp[0].decl).type)) {
		SwigType *ta = SwigType_pop_arrays((yyvsp[0].decl).type);
		if (SwigType_isfunction((yyvsp[0].decl).type)) {
		  Delete(SwigType_pop_function((yyvsp[0].decl).type));
		} else {
		  (yyval.decl).parms = 0;
		}
		SwigType_push((yyvsp[0].decl).type,ta);
		Delete(ta);
	      } else {
		(yyval.decl).parms = 0;
	      }
            }
#line 9098 "CParse/parser.c"
    break;

  case 302: /* plain_declarator: direct_declarator LPAREN parms RPAREN cv_ref_qualifier  */
#line 5380 "./CParse/parser.y"
                                                                     {
	      SwigType *t;
	      (yyval.decl) = (yyvsp[-4].decl);
	      t = NewStringEmpty();
	      SwigType_add_function(t, (yyvsp[-2].pl));
	      if ((yyvsp[0].dtype).qualifier)
	        SwigType_push(t, (yyvsp[0].dtype).qualifier);
	      if (!(yyval.decl).have_parms) {
		(yyval.decl).parms = (yyvsp[-2].pl);
		(yyval.decl).have_parms = 1;
	      }
	      if (!(yyval.decl).type) {
		(yyval.decl).type = t;
	      } else {
		SwigType_push(t, (yyval.decl).type);
		Delete((yyval.decl).type);
		(yyval.decl).type = t;
	      }
	    }
#line 9122 "CParse/parser.c"
    break;

  case 303: /* plain_declarator: %empty  */
#line 5399 "./CParse/parser.y"
                     {
   	      (yyval.decl).type = 0;
              (yyval.decl).id = 0;
	      (yyval.decl).parms = 0;
	      }
#line 9132 "CParse/parser.c"
    break;

  case 304: /* declarator: pointer notso_direct_declarator  */
#line 5406 "./CParse/parser.y"
                                              {
              (yyval.decl) = (yyvsp[0].decl);
	      if ((yyval.decl).type) {
		SwigType_push((yyvsp[-1].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-1].type);
           }
#line 9145 "CParse/parser.c"
    break;

  case 305: /* declarator: pointer AND notso_direct_declarator  */
#line 5414 "./CParse/parser.y"
                                                 {
              (yyval.decl) = (yyvsp[0].decl);
	      SwigType_add_reference((yyvsp[-2].type));
              if ((yyval.decl).type) {
		SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-2].type);
           }
#line 9159 "CParse/parser.c"
    break;

  case 306: /* declarator: pointer LAND notso_direct_declarator  */
#line 5423 "./CParse/parser.y"
                                                  {
              (yyval.decl) = (yyvsp[0].decl);
	      SwigType_add_rvalue_reference((yyvsp[-2].type));
              if ((yyval.decl).type) {
		SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-2].type);
           }
#line 9173 "CParse/parser.c"
    break;

  case 307: /* declarator: direct_declarator  */
#line 5432 "./CParse/parser.y"
                               {
              (yyval.decl) = (yyvsp[0].decl);
	      if (!(yyval.decl).type) (yyval.decl).type = NewStringEmpty();
           }
#line 9182 "CParse/parser.c"
    break;

  case 308: /* declarator: AND notso_direct_declarator  */
#line 5436 "./CParse/parser.y"
                                         {
	     (yyval.decl) = (yyvsp[0].decl);
	     (yyval.decl).type = NewStringEmpty();
	     SwigType_add_reference((yyval.decl).type);
	     if ((yyvsp[0].decl).type) {
	       SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
	       Delete((yyvsp[0].decl).type);
	     }
           }
#line 9196 "CParse/parser.c"
    break;

  case 309: /* declarator: LAND notso_direct_declarator  */
#line 5445 "./CParse/parser.y"
                                          {
	     /* Introduced in C++11, move operator && */
             /* Adds one S/R conflict */
	     (yyval.decl) = (yyvsp[0].decl);
	     (yyval.decl).type = NewStringEmpty();
	     SwigType_add_rvalue_reference((yyval.decl).type);
	     if ((yyvsp[0].decl).type) {
	       SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
	       Delete((yyvsp[0].decl).type);
	     }
           }
#line 9212 "CParse/parser.c"
    break;

  case 310: /* declarator: idcolon DSTAR notso_direct_declarator  */
#line 5456 "./CParse/parser.y"
                                                   { 
	     SwigType *t = NewStringEmpty();

	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-2].str));
	     if ((yyval.decl).type) {
	       SwigType_push(t,(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = t;
	     }
#line 9228 "CParse/parser.c"
    break;

  case 311: /* declarator: pointer idcolon DSTAR notso_direct_declarator  */
#line 5467 "./CParse/parser.y"
                                                           { 
	     SwigType *t = NewStringEmpty();
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-2].str));
	     SwigType_push((yyvsp[-3].type),t);
	     if ((yyval.decl).type) {
	       SwigType_push((yyvsp[-3].type),(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = (yyvsp[-3].type);
	     Delete(t);
	   }
#line 9245 "CParse/parser.c"
    break;

  case 312: /* declarator: pointer idcolon DSTAR AND notso_direct_declarator  */
#line 5479 "./CParse/parser.y"
                                                               { 
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer((yyvsp[-4].type),(yyvsp[-3].str));
	     SwigType_add_reference((yyvsp[-4].type));
	     if ((yyval.decl).type) {
	       SwigType_push((yyvsp[-4].type),(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = (yyvsp[-4].type);
	   }
#line 9260 "CParse/parser.c"
    break;

  case 313: /* declarator: idcolon DSTAR AND notso_direct_declarator  */
#line 5489 "./CParse/parser.y"
                                                       { 
	     SwigType *t = NewStringEmpty();
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-3].str));
	     SwigType_add_reference(t);
	     if ((yyval.decl).type) {
	       SwigType_push(t,(yyval.decl).type);
	       Delete((yyval.decl).type);
	     } 
	     (yyval.decl).type = t;
	   }
#line 9276 "CParse/parser.c"
    break;

  case 314: /* declarator: pointer ELLIPSIS notso_direct_declarator  */
#line 5503 "./CParse/parser.y"
                                                       {
              (yyval.decl) = (yyvsp[0].decl);
	      if ((yyval.decl).type) {
		SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-2].type);
	      SwigType_add_variadic((yyval.decl).type);
           }
#line 9290 "CParse/parser.c"
    break;

  case 315: /* declarator: pointer AND ELLIPSIS notso_direct_declarator  */
#line 5512 "./CParse/parser.y"
                                                          {
              (yyval.decl) = (yyvsp[0].decl);
	      SwigType_add_reference((yyvsp[-3].type));
              if ((yyval.decl).type) {
		SwigType_push((yyvsp[-3].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-3].type);
	      SwigType_add_variadic((yyval.decl).type);
           }
#line 9305 "CParse/parser.c"
    break;

  case 316: /* declarator: pointer LAND ELLIPSIS notso_direct_declarator  */
#line 5522 "./CParse/parser.y"
                                                           {
              (yyval.decl) = (yyvsp[0].decl);
	      SwigType_add_rvalue_reference((yyvsp[-3].type));
              if ((yyval.decl).type) {
		SwigType_push((yyvsp[-3].type),(yyval.decl).type);
		Delete((yyval.decl).type);
	      }
	      (yyval.decl).type = (yyvsp[-3].type);
	      SwigType_add_variadic((yyval.decl).type);
           }
#line 9320 "CParse/parser.c"
    break;

  case 317: /* declarator: AND ELLIPSIS notso_direct_declarator  */
#line 5532 "./CParse/parser.y"
                                                  {
	     (yyval.decl) = (yyvsp[0].decl);
	     (yyval.decl).type = NewStringEmpty();
	     SwigType_add_reference((yyval.decl).type);
	     SwigType_add_variadic((yyval.decl).type);
	     if ((yyvsp[0].decl).type) {
	       SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
	       Delete((yyvsp[0].decl).type);
	     }
           }
#line 9335 "CParse/parser.c"
    break;

  case 318: /* declarator: LAND ELLIPSIS notso_direct_declarator  */
#line 5542 "./CParse/parser.y"
                                                   {
	     /* Introduced in C++11, move operator && */
             /* Adds one S/R conflict */
	     (yyval.decl) = (yyvsp[0].decl);
	     (yyval.decl).type = NewStringEmpty();
	     SwigType_add_rvalue_reference((yyval.decl).type);
	     SwigType_add_variadic((yyval.decl).type);
	     if ((yyvsp[0].decl).type) {
	       SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
	       Delete((yyvsp[0].decl).type);
	     }
           }
#line 9352 "CParse/parser.c"
    break;

  case 319: /* declarator: idcolon DSTAR ELLIPSIS notso_direct_declarator  */
#line 5554 "./CParse/parser.y"
                                                            {
	     SwigType *t = NewStringEmpty();

	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-3].str));
	     SwigType_add_variadic(t);
	     if ((yyval.decl).type) {
	       SwigType_push(t,(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = t;
	     }
#line 9369 "CParse/parser.c"
    break;

  case 320: /* declarator: pointer idcolon DSTAR ELLIPSIS notso_direct_declarator  */
#line 5566 "./CParse/parser.y"
                                                                    {
	     SwigType *t = NewStringEmpty();
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-3].str));
	     SwigType_add_variadic(t);
	     SwigType_push((yyvsp[-4].type),t);
	     if ((yyval.decl).type) {
	       SwigType_push((yyvsp[-4].type),(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = (yyvsp[-4].type);
	     Delete(t);
	   }
#line 9387 "CParse/parser.c"
    break;

  case 321: /* declarator: pointer idcolon DSTAR AND ELLIPSIS notso_direct_declarator  */
#line 5579 "./CParse/parser.y"
                                                                        {
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer((yyvsp[-5].type),(yyvsp[-4].str));
	     SwigType_add_reference((yyvsp[-5].type));
	     SwigType_add_variadic((yyvsp[-5].type));
	     if ((yyval.decl).type) {
	       SwigType_push((yyvsp[-5].type),(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = (yyvsp[-5].type);
	   }
#line 9403 "CParse/parser.c"
    break;

  case 322: /* declarator: pointer idcolon DSTAR LAND ELLIPSIS notso_direct_declarator  */
#line 5590 "./CParse/parser.y"
                                                                         {
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer((yyvsp[-5].type),(yyvsp[-4].str));
	     SwigType_add_rvalue_reference((yyvsp[-5].type));
	     SwigType_add_variadic((yyvsp[-5].type));
	     if ((yyval.decl).type) {
	       SwigType_push((yyvsp[-5].type),(yyval.decl).type);
	       Delete((yyval.decl).type);
	     }
	     (yyval.decl).type = (yyvsp[-5].type);
	   }
#line 9419 "CParse/parser.c"
    break;

  case 323: /* declarator: idcolon DSTAR AND ELLIPSIS notso_direct_declarator  */
#line 5601 "./CParse/parser.y"
                                                                {
	     SwigType *t = NewStringEmpty();
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-4].str));
	     SwigType_add_reference(t);
	     SwigType_add_variadic(t);
	     if ((yyval.decl).type) {
	       SwigType_push(t,(yyval.decl).type);
	       Delete((yyval.decl).type);
	     } 
	     (yyval.decl).type = t;
	   }
#line 9436 "CParse/parser.c"
    break;

  case 324: /* declarator: idcolon DSTAR LAND ELLIPSIS notso_direct_declarator  */
#line 5613 "./CParse/parser.y"
                                                                 {
	     SwigType *t = NewStringEmpty();
	     (yyval.decl) = (yyvsp[0].decl);
	     SwigType_add_memberpointer(t,(yyvsp[-4].str));
	     SwigType_add_rvalue_reference(t);
	     SwigType_add_variadic(t);
	     if ((yyval.decl).type) {
	       SwigType_push(t,(yyval.decl).type);
	       Delete((yyval.decl).type);
	     } 
	     (yyval.decl).type = t;
	   }
#line 9453 "CParse/parser.c"
    break;

  case 325: /* notso_direct_declarator: idcolon  */
#line 5627 "./CParse/parser.y"
                                  {
                /* Note: This is non-standard C.  Template declarator is allowed to follow an identifier */
                 (yyval.decl).id = Char((yyvsp[0].str));
		 (yyval.decl).type = 0;
		 (yyval.decl).parms = 0;
		 (yyval.decl).have_parms = 0;
                  }
#line 9465 "CParse/parser.c"
    break;

  case 326: /* notso_direct_declarator: NOT idcolon  */
#line 5634 "./CParse/parser.y"
                                {
                  (yyval.decl).id = Char(NewStringf("~%s",(yyvsp[0].str)));
                  (yyval.decl).type = 0;
                  (yyval.decl).parms = 0;
                  (yyval.decl).have_parms = 0;
                  }
#line 9476 "CParse/parser.c"
    break;

  case 327: /* notso_direct_declarator: LPAREN idcolon RPAREN  */
#line 5642 "./CParse/parser.y"
                                         {
                  (yyval.decl).id = Char((yyvsp[-1].str));
                  (yyval.decl).type = 0;
                  (yyval.decl).parms = 0;
                  (yyval.decl).have_parms = 0;
                  }
#line 9487 "CParse/parser.c"
    break;

  case 328: /* notso_direct_declarator: LPAREN pointer notso_direct_declarator RPAREN  */
#line 5658 "./CParse/parser.y"
                                                                  {
		    (yyval.decl) = (yyvsp[-1].decl);
		    if ((yyval.decl).type) {
		      SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = (yyvsp[-2].type);
                  }
#line 9500 "CParse/parser.c"
    break;

  case 329: /* notso_direct_declarator: LPAREN idcolon DSTAR notso_direct_declarator RPAREN  */
#line 5666 "./CParse/parser.y"
                                                                        {
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-1].decl);
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t,(yyvsp[-3].str));
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
		    }
#line 9516 "CParse/parser.c"
    break;

  case 330: /* notso_direct_declarator: notso_direct_declarator LBRACKET RBRACKET  */
#line 5677 "./CParse/parser.y"
                                                              { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-2].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,"");
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9532 "CParse/parser.c"
    break;

  case 331: /* notso_direct_declarator: notso_direct_declarator LBRACKET expr RBRACKET  */
#line 5688 "./CParse/parser.y"
                                                                   { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,(yyvsp[-1].dtype).val);
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9548 "CParse/parser.c"
    break;

  case 332: /* notso_direct_declarator: notso_direct_declarator LPAREN parms RPAREN  */
#line 5699 "./CParse/parser.y"
                                                                {
		    SwigType *t;
                    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
		    SwigType_add_function(t,(yyvsp[-1].pl));
		    if (!(yyval.decl).have_parms) {
		      (yyval.decl).parms = (yyvsp[-1].pl);
		      (yyval.decl).have_parms = 1;
		    }
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = t;
		    } else {
		      SwigType_push(t, (yyval.decl).type);
		      Delete((yyval.decl).type);
		      (yyval.decl).type = t;
		    }
		  }
#line 9570 "CParse/parser.c"
    break;

  case 333: /* direct_declarator: idcolon  */
#line 5718 "./CParse/parser.y"
                            {
                /* Note: This is non-standard C.  Template declarator is allowed to follow an identifier */
                 (yyval.decl).id = Char((yyvsp[0].str));
		 (yyval.decl).type = 0;
		 (yyval.decl).parms = 0;
		 (yyval.decl).have_parms = 0;
                  }
#line 9582 "CParse/parser.c"
    break;

  case 334: /* direct_declarator: NOT idcolon  */
#line 5726 "./CParse/parser.y"
                                {
                  (yyval.decl).id = Char(NewStringf("~%s",(yyvsp[0].str)));
                  (yyval.decl).type = 0;
                  (yyval.decl).parms = 0;
                  (yyval.decl).have_parms = 0;
                  }
#line 9593 "CParse/parser.c"
    break;

  case 335: /* direct_declarator: LPAREN pointer direct_declarator RPAREN  */
#line 5743 "./CParse/parser.y"
                                                            {
		    (yyval.decl) = (yyvsp[-1].decl);
		    if ((yyval.decl).type) {
		      SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = (yyvsp[-2].type);
                  }
#line 9606 "CParse/parser.c"
    break;

  case 336: /* direct_declarator: LPAREN AND direct_declarator RPAREN  */
#line 5751 "./CParse/parser.y"
                                                        {
                    (yyval.decl) = (yyvsp[-1].decl);
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = NewStringEmpty();
		    }
		    SwigType_add_reference((yyval.decl).type);
                  }
#line 9618 "CParse/parser.c"
    break;

  case 337: /* direct_declarator: LPAREN LAND direct_declarator RPAREN  */
#line 5758 "./CParse/parser.y"
                                                         {
                    (yyval.decl) = (yyvsp[-1].decl);
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = NewStringEmpty();
		    }
		    SwigType_add_rvalue_reference((yyval.decl).type);
                  }
#line 9630 "CParse/parser.c"
    break;

  case 338: /* direct_declarator: LPAREN idcolon DSTAR declarator RPAREN  */
#line 5765 "./CParse/parser.y"
                                                           {
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-1].decl);
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t,(yyvsp[-3].str));
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
		  }
#line 9646 "CParse/parser.c"
    break;

  case 339: /* direct_declarator: LPAREN idcolon DSTAR type_qualifier declarator RPAREN  */
#line 5776 "./CParse/parser.y"
                                                                          {
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-1].decl);
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t, (yyvsp[-4].str));
		    SwigType_push(t, (yyvsp[-2].str));
		    if ((yyval.decl).type) {
		      SwigType_push(t, (yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
		  }
#line 9663 "CParse/parser.c"
    break;

  case 340: /* direct_declarator: LPAREN idcolon DSTAR abstract_declarator RPAREN  */
#line 5788 "./CParse/parser.y"
                                                                    {
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-1].decl);
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t, (yyvsp[-3].str));
		    if ((yyval.decl).type) {
		      SwigType_push(t, (yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
		  }
#line 9679 "CParse/parser.c"
    break;

  case 341: /* direct_declarator: LPAREN idcolon DSTAR type_qualifier abstract_declarator RPAREN  */
#line 5799 "./CParse/parser.y"
                                                                                   {
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-1].decl);
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t, (yyvsp[-4].str));
		    SwigType_push(t, (yyvsp[-2].str));
		    if ((yyval.decl).type) {
		      SwigType_push(t, (yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
		  }
#line 9696 "CParse/parser.c"
    break;

  case 342: /* direct_declarator: direct_declarator LBRACKET RBRACKET  */
#line 5811 "./CParse/parser.y"
                                                        { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-2].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,"");
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9712 "CParse/parser.c"
    break;

  case 343: /* direct_declarator: direct_declarator LBRACKET expr RBRACKET  */
#line 5822 "./CParse/parser.y"
                                                             { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,(yyvsp[-1].dtype).val);
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9728 "CParse/parser.c"
    break;

  case 344: /* direct_declarator: direct_declarator LPAREN parms RPAREN  */
#line 5833 "./CParse/parser.y"
                                                          {
		    SwigType *t;
                    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
		    SwigType_add_function(t,(yyvsp[-1].pl));
		    if (!(yyval.decl).have_parms) {
		      (yyval.decl).parms = (yyvsp[-1].pl);
		      (yyval.decl).have_parms = 1;
		    }
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = t;
		    } else {
		      SwigType_push(t, (yyval.decl).type);
		      Delete((yyval.decl).type);
		      (yyval.decl).type = t;
		    }
		  }
#line 9750 "CParse/parser.c"
    break;

  case 345: /* direct_declarator: OPERATOR ID LPAREN parms RPAREN  */
#line 5853 "./CParse/parser.y"
                                                   {
		    SwigType *t;
                    Append((yyvsp[-4].str), " "); /* intervening space is mandatory */
                    Append((yyvsp[-4].str), Char((yyvsp[-3].id)));
		    (yyval.decl).id = Char((yyvsp[-4].str));
		    t = NewStringEmpty();
		    SwigType_add_function(t,(yyvsp[-1].pl));
		    if (!(yyval.decl).have_parms) {
		      (yyval.decl).parms = (yyvsp[-1].pl);
		      (yyval.decl).have_parms = 1;
		    }
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = t;
		    } else {
		      SwigType_push(t, (yyval.decl).type);
		      Delete((yyval.decl).type);
		      (yyval.decl).type = t;
		    }
		  }
#line 9774 "CParse/parser.c"
    break;

  case 346: /* abstract_declarator: pointer variadic_opt  */
#line 5874 "./CParse/parser.y"
                                           {
		    (yyval.decl).type = (yyvsp[-1].type);
                    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		    if ((yyvsp[0].str)) SwigType_add_variadic((yyval.decl).type);
                  }
#line 9786 "CParse/parser.c"
    break;

  case 347: /* abstract_declarator: pointer direct_abstract_declarator  */
#line 5881 "./CParse/parser.y"
                                                       { 
                     (yyval.decl) = (yyvsp[0].decl);
                     SwigType_push((yyvsp[-1].type),(yyvsp[0].decl).type);
		     (yyval.decl).type = (yyvsp[-1].type);
		     Delete((yyvsp[0].decl).type);
                  }
#line 9797 "CParse/parser.c"
    break;

  case 348: /* abstract_declarator: pointer AND variadic_opt  */
#line 5887 "./CParse/parser.y"
                                             {
		    (yyval.decl).type = (yyvsp[-2].type);
		    SwigType_add_reference((yyval.decl).type);
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		    if ((yyvsp[0].str)) SwigType_add_variadic((yyval.decl).type);
		  }
#line 9810 "CParse/parser.c"
    break;

  case 349: /* abstract_declarator: pointer LAND variadic_opt  */
#line 5895 "./CParse/parser.y"
                                              {
		    (yyval.decl).type = (yyvsp[-2].type);
		    SwigType_add_rvalue_reference((yyval.decl).type);
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		    if ((yyvsp[0].str)) SwigType_add_variadic((yyval.decl).type);
		  }
#line 9823 "CParse/parser.c"
    break;

  case 350: /* abstract_declarator: pointer AND direct_abstract_declarator  */
#line 5903 "./CParse/parser.y"
                                                           {
		    (yyval.decl) = (yyvsp[0].decl);
		    SwigType_add_reference((yyvsp[-2].type));
		    if ((yyval.decl).type) {
		      SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = (yyvsp[-2].type);
                  }
#line 9837 "CParse/parser.c"
    break;

  case 351: /* abstract_declarator: pointer LAND direct_abstract_declarator  */
#line 5912 "./CParse/parser.y"
                                                            {
		    (yyval.decl) = (yyvsp[0].decl);
		    SwigType_add_rvalue_reference((yyvsp[-2].type));
		    if ((yyval.decl).type) {
		      SwigType_push((yyvsp[-2].type),(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = (yyvsp[-2].type);
                  }
#line 9851 "CParse/parser.c"
    break;

  case 353: /* abstract_declarator: AND direct_abstract_declarator  */
#line 5922 "./CParse/parser.y"
                                                   {
		    (yyval.decl) = (yyvsp[0].decl);
		    (yyval.decl).type = NewStringEmpty();
		    SwigType_add_reference((yyval.decl).type);
		    if ((yyvsp[0].decl).type) {
		      SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
		      Delete((yyvsp[0].decl).type);
		    }
                  }
#line 9865 "CParse/parser.c"
    break;

  case 354: /* abstract_declarator: LAND direct_abstract_declarator  */
#line 5931 "./CParse/parser.y"
                                                    {
		    (yyval.decl) = (yyvsp[0].decl);
		    (yyval.decl).type = NewStringEmpty();
		    SwigType_add_rvalue_reference((yyval.decl).type);
		    if ((yyvsp[0].decl).type) {
		      SwigType_push((yyval.decl).type,(yyvsp[0].decl).type);
		      Delete((yyvsp[0].decl).type);
		    }
                  }
#line 9879 "CParse/parser.c"
    break;

  case 355: /* abstract_declarator: AND variadic_opt  */
#line 5940 "./CParse/parser.y"
                                     {
                    (yyval.decl).id = 0;
                    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
                    (yyval.decl).type = NewStringEmpty();
		    SwigType_add_reference((yyval.decl).type);
		    if ((yyvsp[0].str)) SwigType_add_variadic((yyval.decl).type);
                  }
#line 9892 "CParse/parser.c"
    break;

  case 356: /* abstract_declarator: LAND variadic_opt  */
#line 5948 "./CParse/parser.y"
                                      {
                    (yyval.decl).id = 0;
                    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
                    (yyval.decl).type = NewStringEmpty();
		    SwigType_add_rvalue_reference((yyval.decl).type);
		    if ((yyvsp[0].str)) SwigType_add_variadic((yyval.decl).type);
                  }
#line 9905 "CParse/parser.c"
    break;

  case 357: /* abstract_declarator: idcolon DSTAR  */
#line 5956 "./CParse/parser.y"
                                  { 
		    (yyval.decl).type = NewStringEmpty();
                    SwigType_add_memberpointer((yyval.decl).type,(yyvsp[-1].str));
                    (yyval.decl).id = 0;
                    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
      	          }
#line 9917 "CParse/parser.c"
    break;

  case 358: /* abstract_declarator: idcolon DSTAR type_qualifier  */
#line 5963 "./CParse/parser.y"
                                                 {
		    (yyval.decl).type = NewStringEmpty();
		    SwigType_add_memberpointer((yyval.decl).type, (yyvsp[-2].str));
		    SwigType_push((yyval.decl).type, (yyvsp[0].str));
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		  }
#line 9930 "CParse/parser.c"
    break;

  case 359: /* abstract_declarator: pointer idcolon DSTAR  */
#line 5971 "./CParse/parser.y"
                                          { 
		    SwigType *t = NewStringEmpty();
                    (yyval.decl).type = (yyvsp[-2].type);
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		    SwigType_add_memberpointer(t,(yyvsp[-1].str));
		    SwigType_push((yyval.decl).type,t);
		    Delete(t);
                  }
#line 9945 "CParse/parser.c"
    break;

  case 360: /* abstract_declarator: pointer idcolon DSTAR direct_abstract_declarator  */
#line 5981 "./CParse/parser.y"
                                                                     { 
		    (yyval.decl) = (yyvsp[0].decl);
		    SwigType_add_memberpointer((yyvsp[-3].type),(yyvsp[-2].str));
		    if ((yyval.decl).type) {
		      SwigType_push((yyvsp[-3].type),(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = (yyvsp[-3].type);
                  }
#line 9959 "CParse/parser.c"
    break;

  case 361: /* direct_abstract_declarator: direct_abstract_declarator LBRACKET RBRACKET  */
#line 5992 "./CParse/parser.y"
                                                                          { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-2].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,"");
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9975 "CParse/parser.c"
    break;

  case 362: /* direct_abstract_declarator: direct_abstract_declarator LBRACKET expr RBRACKET  */
#line 6003 "./CParse/parser.y"
                                                                      { 
		    SwigType *t;
		    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
		    SwigType_add_array(t,(yyvsp[-1].dtype).val);
		    if ((yyval.decl).type) {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		    }
		    (yyval.decl).type = t;
                  }
#line 9991 "CParse/parser.c"
    break;

  case 363: /* direct_abstract_declarator: LBRACKET RBRACKET  */
#line 6014 "./CParse/parser.y"
                                      { 
		    (yyval.decl).type = NewStringEmpty();
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		    SwigType_add_array((yyval.decl).type,"");
                  }
#line 10003 "CParse/parser.c"
    break;

  case 364: /* direct_abstract_declarator: LBRACKET expr RBRACKET  */
#line 6021 "./CParse/parser.y"
                                           { 
		    (yyval.decl).type = NewStringEmpty();
		    (yyval.decl).id = 0;
		    (yyval.decl).parms = 0;
		    (yyval.decl).have_parms = 0;
		    SwigType_add_array((yyval.decl).type,(yyvsp[-1].dtype).val);
		  }
#line 10015 "CParse/parser.c"
    break;

  case 365: /* direct_abstract_declarator: LPAREN abstract_declarator RPAREN  */
#line 6028 "./CParse/parser.y"
                                                      {
                    (yyval.decl) = (yyvsp[-1].decl);
		  }
#line 10023 "CParse/parser.c"
    break;

  case 366: /* direct_abstract_declarator: direct_abstract_declarator LPAREN parms RPAREN  */
#line 6031 "./CParse/parser.y"
                                                                   {
		    SwigType *t;
                    (yyval.decl) = (yyvsp[-3].decl);
		    t = NewStringEmpty();
                    SwigType_add_function(t,(yyvsp[-1].pl));
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = t;
		    } else {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		      (yyval.decl).type = t;
		    }
		    if (!(yyval.decl).have_parms) {
		      (yyval.decl).parms = (yyvsp[-1].pl);
		      (yyval.decl).have_parms = 1;
		    }
		  }
#line 10045 "CParse/parser.c"
    break;

  case 367: /* direct_abstract_declarator: direct_abstract_declarator LPAREN parms RPAREN cv_ref_qualifier  */
#line 6048 "./CParse/parser.y"
                                                                                    {
		    SwigType *t;
                    (yyval.decl) = (yyvsp[-4].decl);
		    t = NewStringEmpty();
                    SwigType_add_function(t,(yyvsp[-2].pl));
		    SwigType_push(t, (yyvsp[0].dtype).qualifier);
		    if (!(yyval.decl).type) {
		      (yyval.decl).type = t;
		    } else {
		      SwigType_push(t,(yyval.decl).type);
		      Delete((yyval.decl).type);
		      (yyval.decl).type = t;
		    }
		    if (!(yyval.decl).have_parms) {
		      (yyval.decl).parms = (yyvsp[-2].pl);
		      (yyval.decl).have_parms = 1;
		    }
		  }
#line 10068 "CParse/parser.c"
    break;

  case 368: /* direct_abstract_declarator: LPAREN parms RPAREN  */
#line 6066 "./CParse/parser.y"
                                        {
                    (yyval.decl).type = NewStringEmpty();
                    SwigType_add_function((yyval.decl).type,(yyvsp[-1].pl));
		    (yyval.decl).parms = (yyvsp[-1].pl);
		    (yyval.decl).have_parms = 1;
		    (yyval.decl).id = 0;
                  }
#line 10080 "CParse/parser.c"
    break;

  case 369: /* pointer: STAR type_qualifier pointer  */
#line 6076 "./CParse/parser.y"
                                         { 
             (yyval.type) = NewStringEmpty();
             SwigType_add_pointer((yyval.type));
	     SwigType_push((yyval.type),(yyvsp[-1].str));
	     SwigType_push((yyval.type),(yyvsp[0].type));
	     Delete((yyvsp[0].type));
           }
#line 10092 "CParse/parser.c"
    break;

  case 370: /* pointer: STAR pointer  */
#line 6083 "./CParse/parser.y"
                          {
	     (yyval.type) = NewStringEmpty();
	     SwigType_add_pointer((yyval.type));
	     SwigType_push((yyval.type),(yyvsp[0].type));
	     Delete((yyvsp[0].type));
	   }
#line 10103 "CParse/parser.c"
    break;

  case 371: /* pointer: STAR type_qualifier  */
#line 6089 "./CParse/parser.y"
                                 { 
	     (yyval.type) = NewStringEmpty();
	     SwigType_add_pointer((yyval.type));
	     SwigType_push((yyval.type),(yyvsp[0].str));
           }
#line 10113 "CParse/parser.c"
    break;

  case 372: /* pointer: STAR  */
#line 6094 "./CParse/parser.y"
                  {
	     (yyval.type) = NewStringEmpty();
	     SwigType_add_pointer((yyval.type));
           }
#line 10122 "CParse/parser.c"
    break;

  case 373: /* cv_ref_qualifier: type_qualifier  */
#line 6101 "./CParse/parser.y"
                                  {
		  (yyval.dtype).qualifier = (yyvsp[0].str);
		  (yyval.dtype).refqualifier = 0;
	       }
#line 10131 "CParse/parser.c"
    break;

  case 374: /* cv_ref_qualifier: type_qualifier ref_qualifier  */
#line 6105 "./CParse/parser.y"
                                              {
		  (yyval.dtype).qualifier = (yyvsp[-1].str);
		  (yyval.dtype).refqualifier = (yyvsp[0].str);
		  SwigType_push((yyval.dtype).qualifier, (yyvsp[0].str));
	       }
#line 10141 "CParse/parser.c"
    break;

  case 375: /* cv_ref_qualifier: ref_qualifier  */
#line 6110 "./CParse/parser.y"
                               {
		  (yyval.dtype).qualifier = NewStringEmpty();
		  (yyval.dtype).refqualifier = (yyvsp[0].str);
		  SwigType_push((yyval.dtype).qualifier, (yyvsp[0].str));
	       }
#line 10151 "CParse/parser.c"
    break;

  case 376: /* ref_qualifier: AND  */
#line 6117 "./CParse/parser.y"
                    {
	          (yyval.str) = NewStringEmpty();
	          SwigType_add_reference((yyval.str));
	       }
#line 10160 "CParse/parser.c"
    break;

  case 377: /* ref_qualifier: LAND  */
#line 6121 "./CParse/parser.y"
                      {
	          (yyval.str) = NewStringEmpty();
	          SwigType_add_rvalue_reference((yyval.str));
	       }
#line 10169 "CParse/parser.c"
    break;

  case 378: /* type_qualifier: type_qualifier_raw  */
#line 6127 "./CParse/parser.y"
                                    {
	          (yyval.str) = NewStringEmpty();
	          if ((yyvsp[0].id)) SwigType_add_qualifier((yyval.str),(yyvsp[0].id));
               }
#line 10178 "CParse/parser.c"
    break;

  case 379: /* type_qualifier: type_qualifier_raw type_qualifier  */
#line 6131 "./CParse/parser.y"
                                                   {
		  (yyval.str) = (yyvsp[0].str);
	          if ((yyvsp[-1].id)) SwigType_add_qualifier((yyval.str),(yyvsp[-1].id));
               }
#line 10187 "CParse/parser.c"
    break;

  case 380: /* type_qualifier_raw: CONST_QUAL  */
#line 6137 "./CParse/parser.y"
                                 { (yyval.id) = "const"; }
#line 10193 "CParse/parser.c"
    break;

  case 381: /* type_qualifier_raw: VOLATILE  */
#line 6138 "./CParse/parser.y"
                               { (yyval.id) = "volatile"; }
#line 10199 "CParse/parser.c"
    break;

  case 382: /* type_qualifier_raw: REGISTER  */
#line 6139 "./CParse/parser.y"
                               { (yyval.id) = 0; }
#line 10205 "CParse/parser.c"
    break;

  case 383: /* type: rawtype  */
#line 6145 "./CParse/parser.y"
                                    {
                   (yyval.type) = (yyvsp[0].type);
                   Replace((yyval.type),"typename ","", DOH_REPLACE_ANY);
                }
#line 10214 "CParse/parser.c"
    break;

  case 384: /* rawtype: type_qualifier type_right  */
#line 6151 "./CParse/parser.y"
                                           {
                   (yyval.type) = (yyvsp[0].type);
	           SwigType_push((yyval.type),(yyvsp[-1].str));
               }
#line 10223 "CParse/parser.c"
    break;

  case 386: /* rawtype: type_right type_qualifier  */
#line 6156 "./CParse/parser.y"
                                           {
		  (yyval.type) = (yyvsp[-1].type);
	          SwigType_push((yyval.type),(yyvsp[0].str));
	       }
#line 10232 "CParse/parser.c"
    break;

  case 387: /* rawtype: type_qualifier type_right type_qualifier  */
#line 6160 "./CParse/parser.y"
                                                          {
		  (yyval.type) = (yyvsp[-1].type);
	          SwigType_push((yyval.type),(yyvsp[0].str));
	          SwigType_push((yyval.type),(yyvsp[-2].str));
	       }
#line 10242 "CParse/parser.c"
    break;

  case 388: /* rawtype: rawtype ELLIPSIS  */
#line 6165 "./CParse/parser.y"
                                  {
		  (yyval.type) = (yyvsp[-1].type);
		  SwigType_add_variadic((yyval.type));
	       }
#line 10251 "CParse/parser.c"
    break;

  case 392: /* type_right: c_enum_key idcolon  */
#line 6174 "./CParse/parser.y"
                                    { (yyval.type) = NewStringf("enum %s", (yyvsp[0].str)); }
#line 10257 "CParse/parser.c"
    break;

  case 394: /* type_right: idcolon  */
#line 6177 "./CParse/parser.y"
                                   {
		  (yyval.type) = (yyvsp[0].str);
               }
#line 10265 "CParse/parser.c"
    break;

  case 395: /* type_right: cpptype idcolon  */
#line 6180 "./CParse/parser.y"
                                           {
		 (yyval.type) = NewStringf("%s %s", (yyvsp[-1].type), (yyvsp[0].str));
               }
#line 10273 "CParse/parser.c"
    break;

  case 397: /* @12: %empty  */
#line 6186 "./CParse/parser.y"
                                      {
		 (yyval.str) = get_raw_text_balanced('(', ')');
	       }
#line 10281 "CParse/parser.c"
    break;

  case 398: /* decltype: DECLTYPE LPAREN @12 decltypeexpr  */
#line 6188 "./CParse/parser.y"
                              {
		 String *expr = (yyvsp[-1].str);
		 if ((yyvsp[0].type)) {
		   (yyval.type) = (yyvsp[0].type);
		 } else {
		   (yyval.type) = NewStringf("decltype%s", expr);
		   /* expr includes parentheses but don't include them in the warning message. */
		   Delitem(expr, 0);
		   Delitem(expr, DOH_END);
		   Swig_warning(WARN_CPP11_DECLTYPE, cparse_file, cparse_line, "Unable to deduce decltype for '%s'.\n", expr);
		 }
		 Delete(expr);
	       }
#line 10299 "CParse/parser.c"
    break;

  case 399: /* decltypeexpr: expr RPAREN  */
#line 6203 "./CParse/parser.y"
                             {
		 (yyval.type) = deduce_type(&(yyvsp[-1].dtype));
	       }
#line 10307 "CParse/parser.c"
    break;

  case 400: /* decltypeexpr: error RPAREN  */
#line 6206 "./CParse/parser.y"
                              {
		 /* Avoid a parse error if we can't parse the expression
		  * decltype() is applied to.
		  *
		  * Set $$ to 0 here to trigger the decltype rule above to
		  * issue a warning.
		  */
		 (yyval.type) = 0;
		 if (skip_balanced('(',')') < 0) Exit(EXIT_FAILURE);
		 Clear(scanner_ccode);
	       }
#line 10323 "CParse/parser.c"
    break;

  case 401: /* primitive_type: primitive_type_list  */
#line 6219 "./CParse/parser.y"
                                     {
		 String *type = (yyvsp[0].ptype).type;
		 if (!type) type = NewString("int");
		 if ((yyvsp[0].ptype).us) {
		   (yyval.type) = NewStringf("%s %s", (yyvsp[0].ptype).us, type);
		   Delete((yyvsp[0].ptype).us);
                   Delete(type);
		 } else {
                   (yyval.type) = type;
		 }
		 if (Cmp((yyval.type),"signed int") == 0) {
		   Delete((yyval.type));
		   (yyval.type) = NewString("int");
                 } else if (Cmp((yyval.type),"signed long") == 0) {
		   Delete((yyval.type));
                   (yyval.type) = NewString("long");
                 } else if (Cmp((yyval.type),"signed short") == 0) {
		   Delete((yyval.type));
		   (yyval.type) = NewString("short");
		 } else if (Cmp((yyval.type),"signed long long") == 0) {
		   Delete((yyval.type));
		   (yyval.type) = NewString("long long");
		 }
               }
#line 10352 "CParse/parser.c"
    break;

  case 403: /* primitive_type_list: type_specifier primitive_type_list  */
#line 6246 "./CParse/parser.y"
                                                    {
                    if ((yyvsp[-1].ptype).us && (yyvsp[0].ptype).us) {
		      Swig_error(cparse_file, cparse_line, "Extra %s specifier.\n", (yyvsp[0].ptype).us);
		    }
                    (yyval.ptype) = (yyvsp[0].ptype);
                    if ((yyvsp[-1].ptype).us) (yyval.ptype).us = (yyvsp[-1].ptype).us;
		    if ((yyvsp[-1].ptype).type) {
		      if (!(yyvsp[0].ptype).type) (yyval.ptype).type = (yyvsp[-1].ptype).type;
		      else {
			int err = 0;
			if ((Cmp((yyvsp[-1].ptype).type,"long") == 0)) {
			  if ((Cmp((yyvsp[0].ptype).type,"long") == 0) || (Strncmp((yyvsp[0].ptype).type,"double",6) == 0)) {
			    (yyval.ptype).type = NewStringf("long %s", (yyvsp[0].ptype).type);
			  } else if (Cmp((yyvsp[0].ptype).type,"int") == 0) {
			    (yyval.ptype).type = (yyvsp[-1].ptype).type;
			  } else {
			    err = 1;
			  }
			} else if ((Cmp((yyvsp[-1].ptype).type,"short")) == 0) {
			  if (Cmp((yyvsp[0].ptype).type,"int") == 0) {
			    (yyval.ptype).type = (yyvsp[-1].ptype).type;
			  } else {
			    err = 1;
			  }
			} else if (Cmp((yyvsp[-1].ptype).type,"int") == 0) {
			  (yyval.ptype).type = (yyvsp[0].ptype).type;
			} else if (Cmp((yyvsp[-1].ptype).type,"double") == 0) {
			  if (Cmp((yyvsp[0].ptype).type,"long") == 0) {
			    (yyval.ptype).type = NewString("long double");
			  } else if (Cmp((yyvsp[0].ptype).type,"_Complex") == 0) {
			    (yyval.ptype).type = NewString("double _Complex");
			  } else {
			    err = 1;
			  }
			} else if (Cmp((yyvsp[-1].ptype).type,"float") == 0) {
			  if (Cmp((yyvsp[0].ptype).type,"_Complex") == 0) {
			    (yyval.ptype).type = NewString("float _Complex");
			  } else {
			    err = 1;
			  }
			} else if (Cmp((yyvsp[-1].ptype).type,"_Complex") == 0) {
			  (yyval.ptype).type = NewStringf("%s _Complex", (yyvsp[0].ptype).type);
			} else {
			  err = 1;
			}
			if (err) {
			  Swig_error(cparse_file, cparse_line, "Extra %s specifier.\n", (yyvsp[-1].ptype).type);
			}
		      }
		    }
               }
#line 10408 "CParse/parser.c"
    break;

  case 404: /* type_specifier: TYPE_INT  */
#line 6300 "./CParse/parser.y"
                          { 
		    (yyval.ptype).type = NewString("int");
                    (yyval.ptype).us = 0;
               }
#line 10417 "CParse/parser.c"
    break;

  case 405: /* type_specifier: TYPE_SHORT  */
#line 6304 "./CParse/parser.y"
                            { 
                    (yyval.ptype).type = NewString("short");
                    (yyval.ptype).us = 0;
                }
#line 10426 "CParse/parser.c"
    break;

  case 406: /* type_specifier: TYPE_LONG  */
#line 6308 "./CParse/parser.y"
                           { 
                    (yyval.ptype).type = NewString("long");
                    (yyval.ptype).us = 0;
                }
#line 10435 "CParse/parser.c"
    break;

  case 407: /* type_specifier: TYPE_CHAR  */
#line 6312 "./CParse/parser.y"
                           { 
                    (yyval.ptype).type = NewString("char");
                    (yyval.ptype).us = 0;
                }
#line 10444 "CParse/parser.c"
    break;

  case 408: /* type_specifier: TYPE_WCHAR  */
#line 6316 "./CParse/parser.y"
                            { 
                    (yyval.ptype).type = NewString("wchar_t");
                    (yyval.ptype).us = 0;
                }
#line 10453 "CParse/parser.c"
    break;

  case 409: /* type_specifier: TYPE_FLOAT  */
#line 6320 "./CParse/parser.y"
                            { 
                    (yyval.ptype).type = NewString("float");
                    (yyval.ptype).us = 0;
                }
#line 10462 "CParse/parser.c"
    break;

  case 410: /* type_specifier: TYPE_DOUBLE  */
#line 6324 "./CParse/parser.y"
                             { 
                    (yyval.ptype).type = NewString("double");
                    (yyval.ptype).us = 0;
                }
#line 10471 "CParse/parser.c"
    break;

  case 411: /* type_specifier: TYPE_SIGNED  */
#line 6328 "./CParse/parser.y"
                             { 
                    (yyval.ptype).us = NewString("signed");
                    (yyval.ptype).type = 0;
                }
#line 10480 "CParse/parser.c"
    break;

  case 412: /* type_specifier: TYPE_UNSIGNED  */
#line 6332 "./CParse/parser.y"
                               { 
                    (yyval.ptype).us = NewString("unsigned");
                    (yyval.ptype).type = 0;
                }
#line 10489 "CParse/parser.c"
    break;

  case 413: /* type_specifier: TYPE_COMPLEX  */
#line 6336 "./CParse/parser.y"
                              { 
                    (yyval.ptype).type = NewString("_Complex");
                    (yyval.ptype).us = 0;
                }
#line 10498 "CParse/parser.c"
    break;

  case 414: /* type_specifier: TYPE_NON_ISO_INT8  */
#line 6340 "./CParse/parser.y"
                                   { 
                    (yyval.ptype).type = NewString("__int8");
                    (yyval.ptype).us = 0;
                }
#line 10507 "CParse/parser.c"
    break;

  case 415: /* type_specifier: TYPE_NON_ISO_INT16  */
#line 6344 "./CParse/parser.y"
                                    { 
                    (yyval.ptype).type = NewString("__int16");
                    (yyval.ptype).us = 0;
                }
#line 10516 "CParse/parser.c"
    break;

  case 416: /* type_specifier: TYPE_NON_ISO_INT32  */
#line 6348 "./CParse/parser.y"
                                    { 
                    (yyval.ptype).type = NewString("__int32");
                    (yyval.ptype).us = 0;
                }
#line 10525 "CParse/parser.c"
    break;

  case 417: /* type_specifier: TYPE_NON_ISO_INT64  */
#line 6352 "./CParse/parser.y"
                                    { 
                    (yyval.ptype).type = NewString("__int64");
                    (yyval.ptype).us = 0;
                }
#line 10534 "CParse/parser.c"
    break;

  case 418: /* definetype: expr  */
#line 6358 "./CParse/parser.y"
                      {
                   (yyval.dtype) = (yyvsp[0].dtype);
		   if ((yyval.dtype).type == T_STRING) {
		     (yyval.dtype).rawval = NewStringf("\"%(escape)s\"",(yyval.dtype).val);
		   } else if ((yyval.dtype).type != T_CHAR && (yyval.dtype).type != T_WSTRING && (yyval.dtype).type != T_WCHAR) {
		     (yyval.dtype).rawval = NewStringf("%s", (yyval.dtype).val);
		   }
		   (yyval.dtype).qualifier = 0;
		   (yyval.dtype).refqualifier = 0;
		   (yyval.dtype).bitfield = 0;
		   (yyval.dtype).throws = 0;
		   (yyval.dtype).throwf = 0;
		   (yyval.dtype).nexcept = 0;
		   (yyval.dtype).final = 0;
                }
#line 10554 "CParse/parser.c"
    break;

  case 422: /* deleted_definition: DELETE_KW  */
#line 6381 "./CParse/parser.y"
                               {
		  (yyval.dtype).val = NewString("delete");
		  (yyval.dtype).rawval = 0;
		  (yyval.dtype).type = T_STRING;
		  (yyval.dtype).unary_arg_type = 0;
		  (yyval.dtype).qualifier = 0;
		  (yyval.dtype).refqualifier = 0;
		  (yyval.dtype).bitfield = 0;
		  (yyval.dtype).throws = 0;
		  (yyval.dtype).throwf = 0;
		  (yyval.dtype).nexcept = 0;
		  (yyval.dtype).final = 0;
		}
#line 10572 "CParse/parser.c"
    break;

  case 423: /* explicit_default: DEFAULT  */
#line 6397 "./CParse/parser.y"
                           {
		  (yyval.dtype).val = NewString("default");
		  (yyval.dtype).rawval = 0;
		  (yyval.dtype).type = T_STRING;
		  (yyval.dtype).unary_arg_type = 0;
		  (yyval.dtype).qualifier = 0;
		  (yyval.dtype).refqualifier = 0;
		  (yyval.dtype).bitfield = 0;
		  (yyval.dtype).throws = 0;
		  (yyval.dtype).throwf = 0;
		  (yyval.dtype).nexcept = 0;
		  (yyval.dtype).final = 0;
		}
#line 10590 "CParse/parser.c"
    break;

  case 425: /* ename: %empty  */
#line 6415 "./CParse/parser.y"
                         { (yyval.id) = 0; }
#line 10596 "CParse/parser.c"
    break;

  case 430: /* enumlist: enumlist_item  */
#line 6434 "./CParse/parser.y"
                                {
		  Setattr((yyvsp[0].node),"_last",(yyvsp[0].node));
		  (yyval.node) = (yyvsp[0].node);
		}
#line 10605 "CParse/parser.c"
    break;

  case 431: /* enumlist: enumlist_item DOXYGENPOSTSTRING  */
#line 6438 "./CParse/parser.y"
                                                  {
		  Setattr((yyvsp[-1].node),"_last",(yyvsp[-1].node));
		  set_comment((yyvsp[-1].node), (yyvsp[0].str));
		  (yyval.node) = (yyvsp[-1].node);
		}
#line 10615 "CParse/parser.c"
    break;

  case 432: /* enumlist: enumlist_item DOXYGENSTRING  */
#line 6443 "./CParse/parser.y"
                                              {
		  Setattr((yyvsp[-1].node), "_last", (yyvsp[-1].node));
		  /* Misplaced doxygen string, attach it to previous parameter, like Doxygen does */
		  set_comment((yyvsp[-1].node), (yyvsp[0].str));
		  (yyval.node) = (yyvsp[-1].node);
		}
#line 10626 "CParse/parser.c"
    break;

  case 433: /* enumlist: enumlist_item COMMA enumlist  */
#line 6449 "./CParse/parser.y"
                                               {
		  if ((yyvsp[0].node)) {
		    set_nextSibling((yyvsp[-2].node), (yyvsp[0].node));
		    Setattr((yyvsp[-2].node),"_last",Getattr((yyvsp[0].node),"_last"));
		    Setattr((yyvsp[0].node),"_last",NULL);
		  } else {
		    Setattr((yyvsp[-2].node),"_last",(yyvsp[-2].node));
		  }
		  (yyval.node) = (yyvsp[-2].node);
		}
#line 10641 "CParse/parser.c"
    break;

  case 434: /* enumlist: enumlist_item DOXYGENPOSTSTRING COMMA enumlist  */
#line 6459 "./CParse/parser.y"
                                                                 {
		  if ((yyvsp[0].node)) {
		    set_nextSibling((yyvsp[-3].node), (yyvsp[0].node));
		    Setattr((yyvsp[-3].node),"_last",Getattr((yyvsp[0].node),"_last"));
		    Setattr((yyvsp[0].node),"_last",NULL);
		  } else {
		    Setattr((yyvsp[-3].node),"_last",(yyvsp[-3].node));
		  }
		  set_comment((yyvsp[-3].node), (yyvsp[-2].str));
		  (yyval.node) = (yyvsp[-3].node);
		}
#line 10657 "CParse/parser.c"
    break;

  case 435: /* enumlist: enumlist_item COMMA DOXYGENPOSTSTRING enumlist  */
#line 6470 "./CParse/parser.y"
                                                                 {
		  if ((yyvsp[0].node)) {
		    set_nextSibling((yyvsp[-3].node), (yyvsp[0].node));
		    Setattr((yyvsp[-3].node),"_last",Getattr((yyvsp[0].node),"_last"));
		    Setattr((yyvsp[0].node),"_last",NULL);
		  } else {
		    Setattr((yyvsp[-3].node),"_last",(yyvsp[-3].node));
		  }
		  set_comment((yyvsp[-3].node), (yyvsp[-1].str));
		  (yyval.node) = (yyvsp[-3].node);
		}
#line 10673 "CParse/parser.c"
    break;

  case 436: /* enumlist: optional_ignored_defines  */
#line 6481 "./CParse/parser.y"
                                           {
		  (yyval.node) = 0;
		}
#line 10681 "CParse/parser.c"
    break;

  case 437: /* enumlist_item: optional_ignored_defines edecl_with_dox optional_ignored_defines  */
#line 6486 "./CParse/parser.y"
                                                                                   {
		  (yyval.node) = (yyvsp[-1].node);
		}
#line 10689 "CParse/parser.c"
    break;

  case 439: /* edecl_with_dox: DOXYGENSTRING edecl  */
#line 6492 "./CParse/parser.y"
                                      {
		  (yyval.node) = (yyvsp[0].node);
		  set_comment((yyvsp[0].node), (yyvsp[-1].str));
		}
#line 10698 "CParse/parser.c"
    break;

  case 440: /* edecl: identifier  */
#line 6498 "./CParse/parser.y"
                             {
		   SwigType *type = NewSwigType(T_INT);
		   (yyval.node) = new_node("enumitem");
		   Setattr((yyval.node),"name",(yyvsp[0].id));
		   Setattr((yyval.node),"type",type);
		   SetFlag((yyval.node),"feature:immutable");
		   Delete(type);
		 }
#line 10711 "CParse/parser.c"
    break;

  case 441: /* edecl: identifier EQUAL etype  */
#line 6506 "./CParse/parser.y"
                                          {
		   SwigType *type = NewSwigType((yyvsp[0].dtype).type == T_BOOL ? T_BOOL : ((yyvsp[0].dtype).type == T_CHAR ? T_CHAR : T_INT));
		   (yyval.node) = new_node("enumitem");
		   Setattr((yyval.node),"name",(yyvsp[-2].id));
		   Setattr((yyval.node),"type",type);
		   SetFlag((yyval.node),"feature:immutable");
		   Setattr((yyval.node),"enumvalue", (yyvsp[0].dtype).val);
		   Setattr((yyval.node),"value",(yyvsp[-2].id));
		   Delete(type);
                 }
#line 10726 "CParse/parser.c"
    break;

  case 442: /* etype: expr  */
#line 6518 "./CParse/parser.y"
                        {
                   (yyval.dtype) = (yyvsp[0].dtype);
		   /* We get T_USER here for a typedef - unfortunately we can't
		    * currently resolve typedefs at this stage of parsing. */
		   if (((yyval.dtype).type != T_INT) && ((yyval.dtype).type != T_UINT) &&
		       ((yyval.dtype).type != T_LONG) && ((yyval.dtype).type != T_ULONG) &&
		       ((yyval.dtype).type != T_LONGLONG) && ((yyval.dtype).type != T_ULONGLONG) &&
		       ((yyval.dtype).type != T_SHORT) && ((yyval.dtype).type != T_USHORT) &&
		       ((yyval.dtype).type != T_SCHAR) && ((yyval.dtype).type != T_UCHAR) &&
		       ((yyval.dtype).type != T_CHAR) && ((yyval.dtype).type != T_BOOL) &&
		       ((yyval.dtype).type != T_UNKNOWN) && ((yyval.dtype).type != T_USER)) {
		     Swig_error(cparse_file,cparse_line,"Type error. Expecting an integral type\n");
		   }
                }
#line 10745 "CParse/parser.c"
    break;

  case 444: /* expr: type  */
#line 6537 "./CParse/parser.y"
                      {
		 Node *n;
		 (yyval.dtype).val = (yyvsp[0].type);
		 (yyval.dtype).type = T_UNKNOWN;
		 (yyval.dtype).unary_arg_type = 0;
		 /* Check if value is in scope */
		 n = Swig_symbol_clookup((yyvsp[0].type),0);
		 if (n) {
                   /* A band-aid for enum values used in expressions. */
                   if (Strcmp(nodeType(n),"enumitem") == 0) {
                     String *q = Swig_symbol_qualified(n);
                     if (q) {
                       (yyval.dtype).val = NewStringf("%s::%s", q, Getattr(n,"name"));
		       (yyval.dtype).type = SwigType_type(Getattr(n, "type"));
                       Delete(q);
                     }
		   } else {
		     SwigType *type = Getattr(n, "type");
		     if (type) {
		       (yyval.dtype).type = SwigType_type(type);
		     }
		   }
		 }
               }
#line 10774 "CParse/parser.c"
    break;

  case 445: /* exprmem: ID ARROW ID  */
#line 6564 "./CParse/parser.y"
                             {
		 (yyval.dtype).val = NewStringf("%s->%s", (yyvsp[-2].id), (yyvsp[0].id));
		 (yyval.dtype).type = 0;
	       }
#line 10783 "CParse/parser.c"
    break;

  case 446: /* exprmem: ID ARROW ID LPAREN callparms RPAREN  */
#line 6568 "./CParse/parser.y"
                                                     {
		 (yyval.dtype).val = NewStringf("%s->%s(%s)", (yyvsp[-5].id), (yyvsp[-3].id), (yyvsp[-1].dtype).val);
		 (yyval.dtype).type = 0;
	       }
#line 10792 "CParse/parser.c"
    break;

  case 447: /* exprmem: exprmem ARROW ID  */
#line 6572 "./CParse/parser.y"
                                  {
		 (yyval.dtype) = (yyvsp[-2].dtype);
		 Printf((yyval.dtype).val, "->%s", (yyvsp[0].id));
	       }
#line 10801 "CParse/parser.c"
    break;

  case 448: /* exprmem: exprmem ARROW ID LPAREN callparms RPAREN  */
#line 6576 "./CParse/parser.y"
                                                          {
		 (yyval.dtype) = (yyvsp[-5].dtype);
		 Printf((yyval.dtype).val, "->%s(%s)", (yyvsp[-3].id), (yyvsp[-1].dtype).val);
	       }
#line 10810 "CParse/parser.c"
    break;

  case 449: /* exprmem: ID PERIOD ID  */
#line 6580 "./CParse/parser.y"
                              {
		 (yyval.dtype).val = NewStringf("%s.%s", (yyvsp[-2].id), (yyvsp[0].id));
		 (yyval.dtype).type = 0;
	       }
#line 10819 "CParse/parser.c"
    break;

  case 450: /* exprmem: ID PERIOD ID LPAREN callparms RPAREN  */
#line 6584 "./CParse/parser.y"
                                                      {
		 (yyval.dtype).val = NewStringf("%s.%s(%s)", (yyvsp[-5].id), (yyvsp[-3].id), (yyvsp[-1].dtype).val);
		 (yyval.dtype).type = 0;
	       }
#line 10828 "CParse/parser.c"
    break;

  case 451: /* exprmem: exprmem PERIOD ID  */
#line 6588 "./CParse/parser.y"
                                   {
		 (yyval.dtype) = (yyvsp[-2].dtype);
		 Printf((yyval.dtype).val, ".%s", (yyvsp[0].id));
	       }
#line 10837 "CParse/parser.c"
    break;

  case 452: /* exprmem: exprmem PERIOD ID LPAREN callparms RPAREN  */
#line 6592 "./CParse/parser.y"
                                                           {
		 (yyval.dtype) = (yyvsp[-5].dtype);
		 Printf((yyval.dtype).val, ".%s(%s)", (yyvsp[-3].id), (yyvsp[-1].dtype).val);
	       }
#line 10846 "CParse/parser.c"
    break;

  case 455: /* exprsimple: string  */
#line 6601 "./CParse/parser.y"
                        {
		    (yyval.dtype).val = (yyvsp[0].str);
                    (yyval.dtype).type = T_STRING;
		    (yyval.dtype).unary_arg_type = 0;
               }
#line 10856 "CParse/parser.c"
    break;

  case 456: /* exprsimple: SIZEOF LPAREN type parameter_declarator RPAREN  */
#line 6606 "./CParse/parser.y"
                                                                {
		  SwigType_push((yyvsp[-2].type),(yyvsp[-1].decl).type);
		  (yyval.dtype).val = NewStringf("sizeof(%s)",SwigType_str((yyvsp[-2].type),0));
		  (yyval.dtype).type = T_ULONG;
		  (yyval.dtype).unary_arg_type = 0;
               }
#line 10867 "CParse/parser.c"
    break;

  case 457: /* exprsimple: SIZEOF ELLIPSIS LPAREN type parameter_declarator RPAREN  */
#line 6612 "./CParse/parser.y"
                                                                         {
		  SwigType_push((yyvsp[-2].type),(yyvsp[-1].decl).type);
		  (yyval.dtype).val = NewStringf("sizeof...(%s)",SwigType_str((yyvsp[-2].type),0));
		  (yyval.dtype).type = T_ULONG;
		  (yyval.dtype).unary_arg_type = 0;
               }
#line 10878 "CParse/parser.c"
    break;

  case 458: /* exprsimple: SIZEOF LPAREN exprsimple RPAREN  */
#line 6625 "./CParse/parser.y"
                                                 {
		  (yyval.dtype).val = NewStringf("sizeof(%s)", (yyvsp[-1].dtype).val);
		  (yyval.dtype).type = T_ULONG;
		  (yyval.dtype).unary_arg_type = 0;
	       }
#line 10888 "CParse/parser.c"
    break;

  case 459: /* exprsimple: SIZEOF exprsimple  */
#line 6634 "./CParse/parser.y"
                                   {
		  (yyval.dtype).val = NewStringf("sizeof(%s)", (yyvsp[0].dtype).val);
		  (yyval.dtype).type = T_ULONG;
		  (yyval.dtype).unary_arg_type = 0;
	       }
#line 10898 "CParse/parser.c"
    break;

  case 460: /* exprsimple: wstring  */
#line 6639 "./CParse/parser.y"
                         {
		    (yyval.dtype).val = (yyvsp[0].str);
		    (yyval.dtype).rawval = NewStringf("L\"%s\"", (yyval.dtype).val);
                    (yyval.dtype).type = T_WSTRING;
		    (yyval.dtype).unary_arg_type = 0;
	       }
#line 10909 "CParse/parser.c"
    break;

  case 461: /* exprsimple: CHARCONST  */
#line 6645 "./CParse/parser.y"
                           {
		  (yyval.dtype).val = NewString((yyvsp[0].str));
		  if (Len((yyval.dtype).val)) {
		    (yyval.dtype).rawval = NewStringf("'%(escape)s'", (yyval.dtype).val);
		  } else {
		    (yyval.dtype).rawval = NewString("'\\0'");
		  }
		  (yyval.dtype).type = T_CHAR;
		  (yyval.dtype).unary_arg_type = 0;
		  (yyval.dtype).bitfield = 0;
		  (yyval.dtype).throws = 0;
		  (yyval.dtype).throwf = 0;
		  (yyval.dtype).nexcept = 0;
		  (yyval.dtype).final = 0;
	       }
#line 10929 "CParse/parser.c"
    break;

  case 462: /* exprsimple: WCHARCONST  */
#line 6660 "./CParse/parser.y"
                            {
		  (yyval.dtype).val = NewString((yyvsp[0].str));
		  if (Len((yyval.dtype).val)) {
		    (yyval.dtype).rawval = NewStringf("L\'%s\'", (yyval.dtype).val);
		  } else {
		    (yyval.dtype).rawval = NewString("L'\\0'");
		  }
		  (yyval.dtype).type = T_WCHAR;
		  (yyval.dtype).unary_arg_type = 0;
		  (yyval.dtype).bitfield = 0;
		  (yyval.dtype).throws = 0;
		  (yyval.dtype).throwf = 0;
		  (yyval.dtype).nexcept = 0;
		  (yyval.dtype).final = 0;
	       }
#line 10949 "CParse/parser.c"
    break;

  case 465: /* valexpr: LPAREN expr RPAREN  */
#line 6682 "./CParse/parser.y"
                                                {
		    (yyval.dtype).val = NewStringf("(%s)",(yyvsp[-1].dtype).val);
		    if ((yyvsp[-1].dtype).rawval) {
		      (yyval.dtype).rawval = NewStringf("(%s)",(yyvsp[-1].dtype).rawval);
		    }
		    (yyval.dtype).type = (yyvsp[-1].dtype).type;
	       }
#line 10961 "CParse/parser.c"
    break;

  case 466: /* valexpr: LPAREN expr RPAREN expr  */
#line 6692 "./CParse/parser.y"
                                                    {
		 int cast_type_code = SwigType_type((yyvsp[-2].dtype).val);
		 (yyval.dtype) = (yyvsp[0].dtype);
		 (yyval.dtype).unary_arg_type = 0;
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   switch ((yyvsp[-2].dtype).type) {
		     case T_FLOAT:
		     case T_DOUBLE:
		     case T_LONGDOUBLE:
		     case T_FLTCPLX:
		     case T_DBLCPLX:
		       (yyval.dtype).val = NewStringf("(%s)%s", (yyvsp[-2].dtype).val, (yyvsp[0].dtype).val); /* SwigType_str and decimal points don't mix! */
		       break;
		     default:
		       (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-2].dtype).val,0), (yyvsp[0].dtype).val);
		       break;
		   }
		 }
		 /* As well as C-style casts, this grammar rule currently also
		  * matches a binary operator with a LHS in parentheses for
		  * binary operators which also have an unary form, e.g.:
		  *
		  * (6)*7
		  * (6)&7
		  * (6)+7
		  * (6)-7
		  */
		 if (cast_type_code != T_USER && cast_type_code != T_UNKNOWN) {
		   /* $2 is definitely a type so we know this is a cast. */
		   (yyval.dtype).type = cast_type_code;
		 } else if ((yyvsp[0].dtype).type == 0 || (yyvsp[0].dtype).unary_arg_type == 0) {
		   /* Not one of the cases above, so we know this is a cast. */
		   (yyval.dtype).type = cast_type_code;
		 } else {
		   (yyval.dtype).type = promote((yyvsp[-2].dtype).type, (yyvsp[0].dtype).unary_arg_type);
		 }
 	       }
#line 11003 "CParse/parser.c"
    break;

  case 467: /* valexpr: LPAREN expr pointer RPAREN expr  */
#line 6729 "./CParse/parser.y"
                                                            {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 (yyval.dtype).unary_arg_type = 0;
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   SwigType_push((yyvsp[-3].dtype).val,(yyvsp[-2].type));
		   (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-3].dtype).val,0), (yyvsp[0].dtype).val);
		 }
 	       }
#line 11016 "CParse/parser.c"
    break;

  case 468: /* valexpr: LPAREN expr AND RPAREN expr  */
#line 6737 "./CParse/parser.y"
                                                        {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 (yyval.dtype).unary_arg_type = 0;
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   SwigType_add_reference((yyvsp[-3].dtype).val);
		   (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-3].dtype).val,0), (yyvsp[0].dtype).val);
		 }
 	       }
#line 11029 "CParse/parser.c"
    break;

  case 469: /* valexpr: LPAREN expr LAND RPAREN expr  */
#line 6745 "./CParse/parser.y"
                                                         {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 (yyval.dtype).unary_arg_type = 0;
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   SwigType_add_rvalue_reference((yyvsp[-3].dtype).val);
		   (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-3].dtype).val,0), (yyvsp[0].dtype).val);
		 }
 	       }
#line 11042 "CParse/parser.c"
    break;

  case 470: /* valexpr: LPAREN expr pointer AND RPAREN expr  */
#line 6753 "./CParse/parser.y"
                                                                {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 (yyval.dtype).unary_arg_type = 0;
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   SwigType_push((yyvsp[-4].dtype).val,(yyvsp[-3].type));
		   SwigType_add_reference((yyvsp[-4].dtype).val);
		   (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-4].dtype).val,0), (yyvsp[0].dtype).val);
		 }
 	       }
#line 11056 "CParse/parser.c"
    break;

  case 471: /* valexpr: LPAREN expr pointer LAND RPAREN expr  */
#line 6762 "./CParse/parser.y"
                                                                 {
                 (yyval.dtype) = (yyvsp[0].dtype);
		 (yyval.dtype).unary_arg_type = 0;
		 if ((yyvsp[0].dtype).type != T_STRING) {
		   SwigType_push((yyvsp[-4].dtype).val,(yyvsp[-3].type));
		   SwigType_add_rvalue_reference((yyvsp[-4].dtype).val);
		   (yyval.dtype).val = NewStringf("(%s) %s", SwigType_str((yyvsp[-4].dtype).val,0), (yyvsp[0].dtype).val);
		 }
 	       }
#line 11070 "CParse/parser.c"
    break;

  case 472: /* valexpr: AND expr  */
#line 6771 "./CParse/parser.y"
                          {
		 (yyval.dtype) = (yyvsp[0].dtype);
		 (yyval.dtype).val = NewStringf("&%s", (yyvsp[0].dtype).val);
		 (yyval.dtype).rawval = 0;
		 /* Record the type code for expr so we can properly handle
		  * cases such as (6)&7 which get parsed using this rule then
		  * the rule for a C-style cast.
		  */
		 (yyval.dtype).unary_arg_type = (yyvsp[0].dtype).type;
		 switch ((yyval.dtype).type) {
		   case T_CHAR:
		     (yyval.dtype).type = T_STRING;
		     break;
		   case T_WCHAR:
		     (yyval.dtype).type = T_WSTRING;
		     break;
		   default:
		     (yyval.dtype).type = T_POINTER;
		 }
	       }
#line 11095 "CParse/parser.c"
    break;

  case 473: /* valexpr: STAR expr  */
#line 6791 "./CParse/parser.y"
                           {
		 (yyval.dtype) = (yyvsp[0].dtype);
		 (yyval.dtype).val = NewStringf("*%s", (yyvsp[0].dtype).val);
		 (yyval.dtype).rawval = 0;
		 /* Record the type code for expr so we can properly handle
		  * cases such as (6)*7 which get parsed using this rule then
		  * the rule for a C-style cast.
		  */
		 (yyval.dtype).unary_arg_type = (yyvsp[0].dtype).type;
		 switch ((yyval.dtype).type) {
		   case T_STRING:
		     (yyval.dtype).type = T_CHAR;
		     break;
		   case T_WSTRING:
		     (yyval.dtype).type = T_WCHAR;
		     break;
		   default:
		     (yyval.dtype).type = T_UNKNOWN;
		 }
	       }
#line 11120 "CParse/parser.c"
    break;

  case 484: /* exprcompound: expr PLUS expr  */
#line 6825 "./CParse/parser.y"
                                {
		 (yyval.dtype).val = NewStringf("%s+%s", COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 11129 "CParse/parser.c"
    break;

  case 485: /* exprcompound: expr MINUS expr  */
#line 6829 "./CParse/parser.y"
                                 {
		 (yyval.dtype).val = NewStringf("%s-%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 11138 "CParse/parser.c"
    break;

  case 486: /* exprcompound: expr STAR expr  */
#line 6833 "./CParse/parser.y"
                                {
		 (yyval.dtype).val = NewStringf("%s*%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 11147 "CParse/parser.c"
    break;

  case 487: /* exprcompound: expr SLASH expr  */
#line 6837 "./CParse/parser.y"
                                 {
		 (yyval.dtype).val = NewStringf("%s/%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 11156 "CParse/parser.c"
    break;

  case 488: /* exprcompound: expr MODULO expr  */
#line 6841 "./CParse/parser.y"
                                  {
		 (yyval.dtype).val = NewStringf("%s%%%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 11165 "CParse/parser.c"
    break;

  case 489: /* exprcompound: expr AND expr  */
#line 6845 "./CParse/parser.y"
                               {
		 (yyval.dtype).val = NewStringf("%s&%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 11174 "CParse/parser.c"
    break;

  case 490: /* exprcompound: expr OR expr  */
#line 6849 "./CParse/parser.y"
                              {
		 (yyval.dtype).val = NewStringf("%s|%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 11183 "CParse/parser.c"
    break;

  case 491: /* exprcompound: expr XOR expr  */
#line 6853 "./CParse/parser.y"
                               {
		 (yyval.dtype).val = NewStringf("%s^%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type,(yyvsp[0].dtype).type);
	       }
#line 11192 "CParse/parser.c"
    break;

  case 492: /* exprcompound: expr LSHIFT expr  */
#line 6857 "./CParse/parser.y"
                                  {
		 (yyval.dtype).val = NewStringf("%s << %s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote_type((yyvsp[-2].dtype).type);
	       }
#line 11201 "CParse/parser.c"
    break;

  case 493: /* exprcompound: expr RSHIFT expr  */
#line 6861 "./CParse/parser.y"
                                  {
		 (yyval.dtype).val = NewStringf("%s >> %s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = promote_type((yyvsp[-2].dtype).type);
	       }
#line 11210 "CParse/parser.c"
    break;

  case 494: /* exprcompound: expr LAND expr  */
#line 6865 "./CParse/parser.y"
                                {
		 (yyval.dtype).val = NewStringf("%s&&%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 11219 "CParse/parser.c"
    break;

  case 495: /* exprcompound: expr LOR expr  */
#line 6869 "./CParse/parser.y"
                               {
		 (yyval.dtype).val = NewStringf("%s||%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 11228 "CParse/parser.c"
    break;

  case 496: /* exprcompound: expr EQUALTO expr  */
#line 6873 "./CParse/parser.y"
                                   {
		 (yyval.dtype).val = NewStringf("%s==%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 11237 "CParse/parser.c"
    break;

  case 497: /* exprcompound: expr NOTEQUALTO expr  */
#line 6877 "./CParse/parser.y"
                                      {
		 (yyval.dtype).val = NewStringf("%s!=%s",COMPOUND_EXPR_VAL((yyvsp[-2].dtype)),COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 11246 "CParse/parser.c"
    break;

  case 498: /* exprcompound: LPAREN expr GREATERTHAN expr RPAREN  */
#line 6885 "./CParse/parser.y"
                                                     {
		 (yyval.dtype).val = NewStringf("(%s > %s)", COMPOUND_EXPR_VAL((yyvsp[-3].dtype)), COMPOUND_EXPR_VAL((yyvsp[-1].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 11255 "CParse/parser.c"
    break;

  case 499: /* exprcompound: LPAREN exprsimple LESSTHAN expr RPAREN  */
#line 6895 "./CParse/parser.y"
                                                        {
		 (yyval.dtype).val = NewStringf("(%s < %s)", COMPOUND_EXPR_VAL((yyvsp[-3].dtype)), COMPOUND_EXPR_VAL((yyvsp[-1].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 11264 "CParse/parser.c"
    break;

  case 500: /* exprcompound: expr GREATERTHANOREQUALTO expr  */
#line 6899 "./CParse/parser.y"
                                                {
		 (yyval.dtype).val = NewStringf("%s >= %s", COMPOUND_EXPR_VAL((yyvsp[-2].dtype)), COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 11273 "CParse/parser.c"
    break;

  case 501: /* exprcompound: expr LESSTHANOREQUALTO expr  */
#line 6903 "./CParse/parser.y"
                                             {
		 (yyval.dtype).val = NewStringf("%s <= %s", COMPOUND_EXPR_VAL((yyvsp[-2].dtype)), COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 11282 "CParse/parser.c"
    break;

  case 502: /* exprcompound: expr LESSEQUALGREATER expr  */
#line 6907 "./CParse/parser.y"
                                            {
		 (yyval.dtype).val = NewStringf("%s <=> %s", COMPOUND_EXPR_VAL((yyvsp[-2].dtype)), COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 /* `<=>` returns one of `std::strong_ordering`,
		  * `std::partial_ordering` or `std::weak_ordering`.  The main
		  * thing to do with the return value in this context is to
		  * compare it with another ordering of the same type or
		  * with a literal 0.  We set .type = T_USER here which does
		  * what we want for the comparison operators, and also means
		  * that deduce_type() won't deduce a type for this (which is
		  * better than it deducing the wrong type).
		  */
		 (yyval.dtype).type = T_USER;
		 (yyval.dtype).unary_arg_type = 0;
	       }
#line 11301 "CParse/parser.c"
    break;

  case 503: /* exprcompound: expr QUESTIONMARK expr COLON expr  */
#line 6921 "./CParse/parser.y"
                                                                      {
		 (yyval.dtype).val = NewStringf("%s?%s:%s", COMPOUND_EXPR_VAL((yyvsp[-4].dtype)), COMPOUND_EXPR_VAL((yyvsp[-2].dtype)), COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 /* This may not be exactly right, but is probably good enough
		  * for the purposes of parsing constant expressions. */
		 (yyval.dtype).type = promote((yyvsp[-2].dtype).type, (yyvsp[0].dtype).type);
	       }
#line 11312 "CParse/parser.c"
    break;

  case 504: /* exprcompound: MINUS expr  */
#line 6927 "./CParse/parser.y"
                                         {
		 (yyval.dtype).val = NewStringf("-%s",(yyvsp[0].dtype).val);
		 (yyval.dtype).type = promote_type((yyvsp[0].dtype).type);
		 /* Record the type code for expr so we can properly handle
		  * cases such as (6)-7 which get parsed using this rule then
		  * the rule for a C-style cast.
		  */
		 (yyval.dtype).unary_arg_type = (yyvsp[0].dtype).type;
	       }
#line 11326 "CParse/parser.c"
    break;

  case 505: /* exprcompound: PLUS expr  */
#line 6936 "./CParse/parser.y"
                                        {
                 (yyval.dtype).val = NewStringf("+%s",(yyvsp[0].dtype).val);
		 (yyval.dtype).type = promote_type((yyvsp[0].dtype).type);
		 /* Record the type code for expr so we can properly handle
		  * cases such as (6)+7 which get parsed using this rule then
		  * the rule for a C-style cast.
		  */
		 (yyval.dtype).unary_arg_type = (yyvsp[0].dtype).type;
	       }
#line 11340 "CParse/parser.c"
    break;

  case 506: /* exprcompound: NOT expr  */
#line 6945 "./CParse/parser.y"
                          {
		 (yyval.dtype).val = NewStringf("~%s",(yyvsp[0].dtype).val);
		 (yyval.dtype).type = promote_type((yyvsp[0].dtype).type);
	       }
#line 11349 "CParse/parser.c"
    break;

  case 507: /* exprcompound: LNOT expr  */
#line 6949 "./CParse/parser.y"
                           {
                 (yyval.dtype).val = NewStringf("!%s",COMPOUND_EXPR_VAL((yyvsp[0].dtype)));
		 (yyval.dtype).type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
#line 11358 "CParse/parser.c"
    break;

  case 508: /* exprcompound: type LPAREN  */
#line 6953 "./CParse/parser.y"
                             {
		 String *qty;
		 if (skip_balanced('(',')') < 0) Exit(EXIT_FAILURE);
		 qty = Swig_symbol_type_qualify((yyvsp[-1].type),0);
		 if (SwigType_istemplate(qty)) {
		   String *nstr = SwigType_namestr(qty);
		   Delete(qty);
		   qty = nstr;
		 }
		 (yyval.dtype).val = NewStringf("%s%s",qty,scanner_ccode);
		 Clear(scanner_ccode);
		 /* Try to deduce the type - this could be a C++ "constructor
		  * cast" such as `double(4)` or a function call such as
		  * `some_func()`.  In the latter case we get T_USER, but that
		  * is wrong so we map it to T_UNKNOWN until we can actually
		  * deduce the return type of a function call (which is
		  * complicated because the return type can vary between
		  * overloaded forms).
		  */
		 (yyval.dtype).type = SwigType_type(qty);
		 if ((yyval.dtype).type == T_USER) (yyval.dtype).type = T_UNKNOWN;
		 (yyval.dtype).unary_arg_type = 0;
		 Delete(qty);
               }
#line 11387 "CParse/parser.c"
    break;

  case 509: /* variadic_opt: ELLIPSIS  */
#line 6979 "./CParse/parser.y"
                         {
	        (yyval.str) = NewString("...");
	      }
#line 11395 "CParse/parser.c"
    break;

  case 510: /* variadic_opt: %empty  */
#line 6982 "./CParse/parser.y"
                       {
	        (yyval.str) = 0;
	      }
#line 11403 "CParse/parser.c"
    break;

  case 512: /* $@13: %empty  */
#line 6990 "./CParse/parser.y"
                        { inherit_list = 1; }
#line 11409 "CParse/parser.c"
    break;

  case 513: /* raw_inherit: COLON $@13 base_list  */
#line 6990 "./CParse/parser.y"
                                                        { (yyval.bases) = (yyvsp[0].bases); inherit_list = 0; }
#line 11415 "CParse/parser.c"
    break;

  case 514: /* raw_inherit: %empty  */
#line 6991 "./CParse/parser.y"
                         { (yyval.bases) = 0; }
#line 11421 "CParse/parser.c"
    break;

  case 515: /* base_list: base_specifier  */
#line 6994 "./CParse/parser.y"
                                {
		   Hash *list = NewHash();
		   Node *base = (yyvsp[0].node);
		   Node *name = Getattr(base,"name");
		   List *lpublic = NewList();
		   List *lprotected = NewList();
		   List *lprivate = NewList();
		   Setattr(list,"public",lpublic);
		   Setattr(list,"protected",lprotected);
		   Setattr(list,"private",lprivate);
		   Delete(lpublic);
		   Delete(lprotected);
		   Delete(lprivate);
		   Append(Getattr(list,Getattr(base,"access")),name);
	           (yyval.bases) = list;
               }
#line 11442 "CParse/parser.c"
    break;

  case 516: /* base_list: base_list COMMA base_specifier  */
#line 7011 "./CParse/parser.y"
                                                {
		   Hash *list = (yyvsp[-2].bases);
		   Node *base = (yyvsp[0].node);
		   Node *name = Getattr(base,"name");
		   Append(Getattr(list,Getattr(base,"access")),name);
                   (yyval.bases) = list;
               }
#line 11454 "CParse/parser.c"
    break;

  case 517: /* @14: %empty  */
#line 7020 "./CParse/parser.y"
                                       {
		 (yyval.intvalue) = cparse_line;
	       }
#line 11462 "CParse/parser.c"
    break;

  case 518: /* base_specifier: opt_virtual @14 idcolon variadic_opt  */
#line 7022 "./CParse/parser.y"
                                      {
		 (yyval.node) = NewHash();
		 Setfile((yyval.node),cparse_file);
		 Setline((yyval.node),(yyvsp[-2].intvalue));
		 Setattr((yyval.node),"name",(yyvsp[-1].str));
		 Setfile((yyvsp[-1].str),cparse_file);
		 Setline((yyvsp[-1].str),(yyvsp[-2].intvalue));
                 if (last_cpptype && (Strcmp(last_cpptype,"struct") != 0)) {
		   Setattr((yyval.node),"access","private");
		   Swig_warning(WARN_PARSE_NO_ACCESS, Getfile((yyval.node)), Getline((yyval.node)), "No access specifier given for base class '%s' (ignored).\n", SwigType_namestr((yyvsp[-1].str)));
                 } else {
		   Setattr((yyval.node),"access","public");
		 }
		 if ((yyvsp[0].str)) {
		   SwigType_add_variadic(Getattr((yyval.node), "name"));
		 }
               }
#line 11484 "CParse/parser.c"
    break;

  case 519: /* @15: %empty  */
#line 7039 "./CParse/parser.y"
                                                        {
		 (yyval.intvalue) = cparse_line;
	       }
#line 11492 "CParse/parser.c"
    break;

  case 520: /* base_specifier: opt_virtual access_specifier @15 opt_virtual idcolon variadic_opt  */
#line 7041 "./CParse/parser.y"
                                                  {
		 (yyval.node) = NewHash();
		 Setfile((yyval.node),cparse_file);
		 Setline((yyval.node),(yyvsp[-3].intvalue));
		 Setattr((yyval.node),"name",(yyvsp[-1].str));
		 Setfile((yyvsp[-1].str),cparse_file);
		 Setline((yyvsp[-1].str),(yyvsp[-3].intvalue));
		 Setattr((yyval.node),"access",(yyvsp[-4].id));
	         if (Strcmp((yyvsp[-4].id),"public") != 0) {
		   Swig_warning(WARN_PARSE_PRIVATE_INHERIT, Getfile((yyval.node)), Getline((yyval.node)), "%s inheritance from base '%s' (ignored).\n", (yyvsp[-4].id), SwigType_namestr((yyvsp[-1].str)));
		 }
		 if ((yyvsp[0].str)) {
		   SwigType_add_variadic(Getattr((yyval.node), "name"));
		 }
               }
#line 11512 "CParse/parser.c"
    break;

  case 521: /* access_specifier: PUBLIC  */
#line 7058 "./CParse/parser.y"
                           { (yyval.id) = "public"; }
#line 11518 "CParse/parser.c"
    break;

  case 522: /* access_specifier: PRIVATE  */
#line 7059 "./CParse/parser.y"
                         { (yyval.id) = "private"; }
#line 11524 "CParse/parser.c"
    break;

  case 523: /* access_specifier: PROTECTED  */
#line 7060 "./CParse/parser.y"
                           { (yyval.id) = "protected"; }
#line 11530 "CParse/parser.c"
    break;

  case 524: /* templcpptype: CLASS variadic_opt  */
#line 7063 "./CParse/parser.y"
                                    {
                   (yyval.type) = NewString("class");
		   if (!inherit_list) last_cpptype = (yyval.type);
		   if ((yyvsp[0].str)) SwigType_add_variadic((yyval.type));
               }
#line 11540 "CParse/parser.c"
    break;

  case 525: /* templcpptype: TYPENAME variadic_opt  */
#line 7068 "./CParse/parser.y"
                                       {
                   (yyval.type) = NewString("typename");
		   if (!inherit_list) last_cpptype = (yyval.type);
		   if ((yyvsp[0].str)) SwigType_add_variadic((yyval.type));
               }
#line 11550 "CParse/parser.c"
    break;

  case 527: /* cpptype: STRUCT  */
#line 7076 "./CParse/parser.y"
                        {
                   (yyval.type) = NewString("struct");
		   if (!inherit_list) last_cpptype = (yyval.type);
               }
#line 11559 "CParse/parser.c"
    break;

  case 528: /* cpptype: UNION  */
#line 7080 "./CParse/parser.y"
                       {
                   (yyval.type) = NewString("union");
		   if (!inherit_list) last_cpptype = (yyval.type);
               }
#line 11568 "CParse/parser.c"
    break;

  case 529: /* classkey: CLASS  */
#line 7086 "./CParse/parser.y"
                       {
                   (yyval.type) = NewString("class");
		   if (!inherit_list) last_cpptype = (yyval.type);
               }
#line 11577 "CParse/parser.c"
    break;

  case 530: /* classkey: STRUCT  */
#line 7090 "./CParse/parser.y"
                        {
                   (yyval.type) = NewString("struct");
		   if (!inherit_list) last_cpptype = (yyval.type);
               }
#line 11586 "CParse/parser.c"
    break;

  case 531: /* classkey: UNION  */
#line 7094 "./CParse/parser.y"
                       {
                   (yyval.type) = NewString("union");
		   if (!inherit_list) last_cpptype = (yyval.type);
               }
#line 11595 "CParse/parser.c"
    break;

  case 533: /* classkeyopt: %empty  */
#line 7101 "./CParse/parser.y"
                        {
		   (yyval.type) = 0;
               }
#line 11603 "CParse/parser.c"
    break;

  case 536: /* virt_specifier_seq: OVERRIDE  */
#line 7110 "./CParse/parser.y"
                              {
                   (yyval.str) = 0;
	       }
#line 11611 "CParse/parser.c"
    break;

  case 537: /* virt_specifier_seq: FINAL  */
#line 7113 "./CParse/parser.y"
                       {
                   (yyval.str) = NewString("1");
	       }
#line 11619 "CParse/parser.c"
    break;

  case 538: /* virt_specifier_seq: FINAL OVERRIDE  */
#line 7116 "./CParse/parser.y"
                                {
                   (yyval.str) = NewString("1");
	       }
#line 11627 "CParse/parser.c"
    break;

  case 539: /* virt_specifier_seq: OVERRIDE FINAL  */
#line 7119 "./CParse/parser.y"
                                {
                   (yyval.str) = NewString("1");
	       }
#line 11635 "CParse/parser.c"
    break;

  case 541: /* virt_specifier_seq_opt: %empty  */
#line 7125 "./CParse/parser.y"
                        {
                   (yyval.str) = 0;
               }
#line 11643 "CParse/parser.c"
    break;

  case 542: /* class_virt_specifier_opt: FINAL  */
#line 7130 "./CParse/parser.y"
                                 {
                   (yyval.str) = NewString("1");
               }
#line 11651 "CParse/parser.c"
    break;

  case 543: /* class_virt_specifier_opt: %empty  */
#line 7133 "./CParse/parser.y"
                        {
                   (yyval.str) = 0;
               }
#line 11659 "CParse/parser.c"
    break;

  case 544: /* exception_specification: THROW LPAREN parms RPAREN  */
#line 7138 "./CParse/parser.y"
                                                    {
                    (yyval.dtype).throws = (yyvsp[-1].pl);
                    (yyval.dtype).throwf = NewString("1");
                    (yyval.dtype).nexcept = 0;
                    (yyval.dtype).final = 0;
	       }
#line 11670 "CParse/parser.c"
    break;

  case 545: /* exception_specification: NOEXCEPT  */
#line 7144 "./CParse/parser.y"
                          {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = NewString("true");
                    (yyval.dtype).final = 0;
	       }
#line 11681 "CParse/parser.c"
    break;

  case 546: /* exception_specification: virt_specifier_seq  */
#line 7150 "./CParse/parser.y"
                                    {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = 0;
                    (yyval.dtype).final = (yyvsp[0].str);
	       }
#line 11692 "CParse/parser.c"
    break;

  case 547: /* exception_specification: THROW LPAREN parms RPAREN virt_specifier_seq  */
#line 7156 "./CParse/parser.y"
                                                              {
                    (yyval.dtype).throws = (yyvsp[-2].pl);
                    (yyval.dtype).throwf = NewString("1");
                    (yyval.dtype).nexcept = 0;
                    (yyval.dtype).final = (yyvsp[0].str);
	       }
#line 11703 "CParse/parser.c"
    break;

  case 548: /* exception_specification: NOEXCEPT virt_specifier_seq  */
#line 7162 "./CParse/parser.y"
                                             {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = NewString("true");
                    (yyval.dtype).final = (yyvsp[0].str);
	       }
#line 11714 "CParse/parser.c"
    break;

  case 549: /* exception_specification: NOEXCEPT LPAREN expr RPAREN  */
#line 7168 "./CParse/parser.y"
                                             {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = (yyvsp[-1].dtype).val;
                    (yyval.dtype).final = 0;
	       }
#line 11725 "CParse/parser.c"
    break;

  case 550: /* qualifiers_exception_specification: cv_ref_qualifier  */
#line 7176 "./CParse/parser.y"
                                                      {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = 0;
                    (yyval.dtype).final = 0;
                    (yyval.dtype).qualifier = (yyvsp[0].dtype).qualifier;
                    (yyval.dtype).refqualifier = (yyvsp[0].dtype).refqualifier;
               }
#line 11738 "CParse/parser.c"
    break;

  case 551: /* qualifiers_exception_specification: exception_specification  */
#line 7184 "./CParse/parser.y"
                                         {
		    (yyval.dtype) = (yyvsp[0].dtype);
                    (yyval.dtype).qualifier = 0;
                    (yyval.dtype).refqualifier = 0;
               }
#line 11748 "CParse/parser.c"
    break;

  case 552: /* qualifiers_exception_specification: cv_ref_qualifier exception_specification  */
#line 7189 "./CParse/parser.y"
                                                          {
		    (yyval.dtype) = (yyvsp[0].dtype);
                    (yyval.dtype).qualifier = (yyvsp[-1].dtype).qualifier;
                    (yyval.dtype).refqualifier = (yyvsp[-1].dtype).refqualifier;
               }
#line 11758 "CParse/parser.c"
    break;

  case 554: /* cpp_const: %empty  */
#line 7197 "./CParse/parser.y"
                        {
                    (yyval.dtype).throws = 0;
                    (yyval.dtype).throwf = 0;
                    (yyval.dtype).nexcept = 0;
                    (yyval.dtype).final = 0;
                    (yyval.dtype).qualifier = 0;
                    (yyval.dtype).refqualifier = 0;
               }
#line 11771 "CParse/parser.c"
    break;

  case 555: /* ctor_end: cpp_const ctor_initializer SEMI  */
#line 7207 "./CParse/parser.y"
                                                 { 
                    Clear(scanner_ccode); 
                    (yyval.decl).have_parms = 0; 
                    (yyval.decl).defarg = 0; 
		    (yyval.decl).throws = (yyvsp[-2].dtype).throws;
		    (yyval.decl).throwf = (yyvsp[-2].dtype).throwf;
		    (yyval.decl).nexcept = (yyvsp[-2].dtype).nexcept;
		    (yyval.decl).final = (yyvsp[-2].dtype).final;
                    if ((yyvsp[-2].dtype).qualifier)
                      Swig_error(cparse_file, cparse_line, "Constructor cannot have a qualifier.\n");
               }
#line 11787 "CParse/parser.c"
    break;

  case 556: /* ctor_end: cpp_const ctor_initializer LBRACE  */
#line 7218 "./CParse/parser.y"
                                                   { 
                    if ((yyvsp[-2].dtype).qualifier)
                      Swig_error(cparse_file, cparse_line, "Constructor cannot have a qualifier.\n");
                    if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
                    (yyval.decl).have_parms = 0; 
                    (yyval.decl).defarg = 0; 
                    (yyval.decl).throws = (yyvsp[-2].dtype).throws;
                    (yyval.decl).throwf = (yyvsp[-2].dtype).throwf;
                    (yyval.decl).nexcept = (yyvsp[-2].dtype).nexcept;
                    (yyval.decl).final = (yyvsp[-2].dtype).final;
               }
#line 11803 "CParse/parser.c"
    break;

  case 557: /* ctor_end: LPAREN parms RPAREN SEMI  */
#line 7229 "./CParse/parser.y"
                                          { 
                    Clear(scanner_ccode); 
                    (yyval.decl).parms = (yyvsp[-2].pl); 
                    (yyval.decl).have_parms = 1; 
                    (yyval.decl).defarg = 0; 
		    (yyval.decl).throws = 0;
		    (yyval.decl).throwf = 0;
		    (yyval.decl).nexcept = 0;
		    (yyval.decl).final = 0;
               }
#line 11818 "CParse/parser.c"
    break;

  case 558: /* ctor_end: LPAREN parms RPAREN LBRACE  */
#line 7239 "./CParse/parser.y"
                                            {
                    if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
                    (yyval.decl).parms = (yyvsp[-2].pl); 
                    (yyval.decl).have_parms = 1; 
                    (yyval.decl).defarg = 0; 
                    (yyval.decl).throws = 0;
                    (yyval.decl).throwf = 0;
                    (yyval.decl).nexcept = 0;
                    (yyval.decl).final = 0;
               }
#line 11833 "CParse/parser.c"
    break;

  case 559: /* ctor_end: EQUAL definetype SEMI  */
#line 7249 "./CParse/parser.y"
                                       { 
                    (yyval.decl).have_parms = 0; 
                    (yyval.decl).defarg = (yyvsp[-1].dtype).val; 
                    (yyval.decl).throws = 0;
                    (yyval.decl).throwf = 0;
                    (yyval.decl).nexcept = 0;
                    (yyval.decl).final = 0;
               }
#line 11846 "CParse/parser.c"
    break;

  case 560: /* ctor_end: exception_specification EQUAL default_delete SEMI  */
#line 7257 "./CParse/parser.y"
                                                                   {
                    (yyval.decl).have_parms = 0;
                    (yyval.decl).defarg = (yyvsp[-1].dtype).val;
                    (yyval.decl).throws = (yyvsp[-3].dtype).throws;
                    (yyval.decl).throwf = (yyvsp[-3].dtype).throwf;
                    (yyval.decl).nexcept = (yyvsp[-3].dtype).nexcept;
                    (yyval.decl).final = (yyvsp[-3].dtype).final;
                    if ((yyvsp[-3].dtype).qualifier)
                      Swig_error(cparse_file, cparse_line, "Constructor cannot have a qualifier.\n");
               }
#line 11861 "CParse/parser.c"
    break;

  case 567: /* mem_initializer: idcolon LPAREN  */
#line 7279 "./CParse/parser.y"
                                 {
		  if (skip_balanced('(',')') < 0) Exit(EXIT_FAILURE);
		  Clear(scanner_ccode);
		}
#line 11870 "CParse/parser.c"
    break;

  case 568: /* mem_initializer: idcolon LBRACE  */
#line 7291 "./CParse/parser.y"
                                 {
		  if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		  Clear(scanner_ccode);
		}
#line 11879 "CParse/parser.c"
    break;

  case 569: /* less_valparms_greater: LESSTHAN valparms GREATERTHAN  */
#line 7297 "./CParse/parser.y"
                                                      {
                     String *s = NewStringEmpty();
                     SwigType_add_template(s,(yyvsp[-1].p));
                     (yyval.id) = Char(s);
		     scanner_last_id(1);
                }
#line 11890 "CParse/parser.c"
    break;

  case 571: /* identifier: OVERRIDE  */
#line 7307 "./CParse/parser.y"
                          { (yyval.id) = Swig_copy_string("override"); }
#line 11896 "CParse/parser.c"
    break;

  case 572: /* identifier: FINAL  */
#line 7308 "./CParse/parser.y"
                       { (yyval.id) = Swig_copy_string("final"); }
#line 11902 "CParse/parser.c"
    break;

  case 574: /* idstring: default_delete  */
#line 7312 "./CParse/parser.y"
                                { (yyval.id) = Char((yyvsp[0].dtype).val); }
#line 11908 "CParse/parser.c"
    break;

  case 575: /* idstring: string  */
#line 7313 "./CParse/parser.y"
                        { (yyval.id) = Char((yyvsp[0].str)); }
#line 11914 "CParse/parser.c"
    break;

  case 577: /* idstringopt: %empty  */
#line 7317 "./CParse/parser.y"
                        { (yyval.id) = 0; }
#line 11920 "CParse/parser.c"
    break;

  case 578: /* idcolon: idtemplate idcolontail  */
#line 7320 "./CParse/parser.y"
                                        { 
		 (yyval.str) = NewStringf("%s%s", (yyvsp[-1].str), (yyvsp[0].str));
		 Delete((yyvsp[0].str));
               }
#line 11929 "CParse/parser.c"
    break;

  case 579: /* idcolon: NONID DCOLON idtemplatetemplate idcolontail  */
#line 7324 "./CParse/parser.y"
                                                             {
		 (yyval.str) = NewStringf("::%s%s",(yyvsp[-1].str),(yyvsp[0].str));
                 Delete((yyvsp[0].str));
               }
#line 11938 "CParse/parser.c"
    break;

  case 580: /* idcolon: idtemplate  */
#line 7328 "./CParse/parser.y"
                            {
		 (yyval.str) = NewString((yyvsp[0].str));
   	       }
#line 11946 "CParse/parser.c"
    break;

  case 581: /* idcolon: NONID DCOLON idtemplatetemplate  */
#line 7331 "./CParse/parser.y"
                                                 {
		 (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 11954 "CParse/parser.c"
    break;

  case 582: /* idcolon: OPERATOR  */
#line 7334 "./CParse/parser.y"
                                    {
                 (yyval.str) = NewStringf("%s", (yyvsp[0].str));
	       }
#line 11962 "CParse/parser.c"
    break;

  case 583: /* idcolon: OPERATOR less_valparms_greater  */
#line 7337 "./CParse/parser.y"
                                                {
                 (yyval.str) = NewStringf("%s%s", (yyvsp[-1].str), (yyvsp[0].id));
	       }
#line 11970 "CParse/parser.c"
    break;

  case 584: /* idcolon: NONID DCOLON OPERATOR  */
#line 7340 "./CParse/parser.y"
                                       {
                 (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 11978 "CParse/parser.c"
    break;

  case 585: /* idcolontail: DCOLON idtemplatetemplate idcolontail  */
#line 7345 "./CParse/parser.y"
                                                       {
                   (yyval.str) = NewStringf("::%s%s",(yyvsp[-1].str),(yyvsp[0].str));
		   Delete((yyvsp[0].str));
               }
#line 11987 "CParse/parser.c"
    break;

  case 586: /* idcolontail: DCOLON idtemplatetemplate  */
#line 7349 "./CParse/parser.y"
                                           {
                   (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 11995 "CParse/parser.c"
    break;

  case 587: /* idcolontail: DCOLON OPERATOR  */
#line 7352 "./CParse/parser.y"
                                 {
                   (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 12003 "CParse/parser.c"
    break;

  case 588: /* idcolontail: DCNOT idtemplate  */
#line 7359 "./CParse/parser.y"
                                  {
		 (yyval.str) = NewStringf("::~%s",(yyvsp[0].str));
               }
#line 12011 "CParse/parser.c"
    break;

  case 589: /* idtemplate: identifier  */
#line 7365 "./CParse/parser.y"
                           {
		(yyval.str) = NewStringf("%s", (yyvsp[0].id));
	      }
#line 12019 "CParse/parser.c"
    break;

  case 590: /* idtemplate: identifier less_valparms_greater  */
#line 7368 "./CParse/parser.y"
                                                 {
		(yyval.str) = NewStringf("%s%s", (yyvsp[-1].id), (yyvsp[0].id));
	      }
#line 12027 "CParse/parser.c"
    break;

  case 592: /* idtemplatetemplate: TEMPLATE identifier less_valparms_greater  */
#line 7374 "./CParse/parser.y"
                                                          {
		(yyval.str) = NewStringf("%s%s", (yyvsp[-1].id), (yyvsp[0].id));
	      }
#line 12035 "CParse/parser.c"
    break;

  case 593: /* idcolonnt: identifier idcolontailnt  */
#line 7380 "./CParse/parser.y"
                                         {
		 (yyval.str) = NewStringf("%s%s", (yyvsp[-1].id), (yyvsp[0].str));
		 Delete((yyvsp[0].str));
               }
#line 12044 "CParse/parser.c"
    break;

  case 594: /* idcolonnt: NONID DCOLON identifier idcolontailnt  */
#line 7384 "./CParse/parser.y"
                                                       {
		 (yyval.str) = NewStringf("::%s%s",(yyvsp[-1].id),(yyvsp[0].str));
                 Delete((yyvsp[0].str));
               }
#line 12053 "CParse/parser.c"
    break;

  case 595: /* idcolonnt: identifier  */
#line 7388 "./CParse/parser.y"
                            {
		 (yyval.str) = NewString((yyvsp[0].id));
   	       }
#line 12061 "CParse/parser.c"
    break;

  case 596: /* idcolonnt: NONID DCOLON identifier  */
#line 7391 "./CParse/parser.y"
                                         {
		 (yyval.str) = NewStringf("::%s",(yyvsp[0].id));
               }
#line 12069 "CParse/parser.c"
    break;

  case 597: /* idcolonnt: OPERATOR  */
#line 7394 "./CParse/parser.y"
                          {
                 (yyval.str) = NewString((yyvsp[0].str));
	       }
#line 12077 "CParse/parser.c"
    break;

  case 598: /* idcolonnt: NONID DCOLON OPERATOR  */
#line 7397 "./CParse/parser.y"
                                       {
                 (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 12085 "CParse/parser.c"
    break;

  case 599: /* idcolontailnt: DCOLON identifier idcolontailnt  */
#line 7402 "./CParse/parser.y"
                                                  {
                   (yyval.str) = NewStringf("::%s%s",(yyvsp[-1].id),(yyvsp[0].str));
		   Delete((yyvsp[0].str));
               }
#line 12094 "CParse/parser.c"
    break;

  case 600: /* idcolontailnt: DCOLON identifier  */
#line 7406 "./CParse/parser.y"
                                   {
                   (yyval.str) = NewStringf("::%s",(yyvsp[0].id));
               }
#line 12102 "CParse/parser.c"
    break;

  case 601: /* idcolontailnt: DCOLON OPERATOR  */
#line 7409 "./CParse/parser.y"
                                 {
                   (yyval.str) = NewStringf("::%s",(yyvsp[0].str));
               }
#line 12110 "CParse/parser.c"
    break;

  case 602: /* idcolontailnt: DCNOT identifier  */
#line 7412 "./CParse/parser.y"
                                  {
		 (yyval.str) = NewStringf("::~%s",(yyvsp[0].id));
               }
#line 12118 "CParse/parser.c"
    break;

  case 603: /* string: string STRING  */
#line 7418 "./CParse/parser.y"
                               { 
                   (yyval.str) = NewStringf("%s%s", (yyvsp[-1].str), (yyvsp[0].id));
               }
#line 12126 "CParse/parser.c"
    break;

  case 604: /* string: STRING  */
#line 7421 "./CParse/parser.y"
                        { (yyval.str) = NewString((yyvsp[0].id));}
#line 12132 "CParse/parser.c"
    break;

  case 605: /* wstring: wstring WSTRING  */
#line 7424 "./CParse/parser.y"
                                  {
                   (yyval.str) = NewStringf("%s%s", (yyvsp[-1].str), (yyvsp[0].id));
               }
#line 12140 "CParse/parser.c"
    break;

  case 606: /* wstring: WSTRING  */
#line 7432 "./CParse/parser.y"
                         { (yyval.str) = NewString((yyvsp[0].id));}
#line 12146 "CParse/parser.c"
    break;

  case 608: /* stringbrace: LBRACE  */
#line 7436 "./CParse/parser.y"
                        {
		  if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		  (yyval.str) = NewString(scanner_ccode);
               }
#line 12155 "CParse/parser.c"
    break;

  case 610: /* options: LPAREN kwargs RPAREN  */
#line 7443 "./CParse/parser.y"
                                      {
                  Hash *n;
                  (yyval.node) = NewHash();
                  n = (yyvsp[-1].node);
                  while(n) {
                     String *name, *value;
                     name = Getattr(n,"name");
                     value = Getattr(n,"value");
		     if (!value) value = (String *) "1";
                     Setattr((yyval.node),name, value);
		     n = nextSibling(n);
		  }
               }
#line 12173 "CParse/parser.c"
    break;

  case 611: /* options: %empty  */
#line 7456 "./CParse/parser.y"
                        { (yyval.node) = 0; }
#line 12179 "CParse/parser.c"
    break;

  case 612: /* kwargs: idstring EQUAL stringnum  */
#line 7460 "./CParse/parser.y"
                                          {
		 (yyval.node) = NewHash();
		 Setattr((yyval.node),"name",(yyvsp[-2].id));
		 Setattr((yyval.node),"value",(yyvsp[0].str));
               }
#line 12189 "CParse/parser.c"
    break;

  case 613: /* kwargs: idstring EQUAL stringnum COMMA kwargs  */
#line 7465 "./CParse/parser.y"
                                                       {
		 (yyval.node) = NewHash();
		 Setattr((yyval.node),"name",(yyvsp[-4].id));
		 Setattr((yyval.node),"value",(yyvsp[-2].str));
		 set_nextSibling((yyval.node),(yyvsp[0].node));
               }
#line 12200 "CParse/parser.c"
    break;

  case 614: /* kwargs: idstring  */
#line 7471 "./CParse/parser.y"
                          {
                 (yyval.node) = NewHash();
                 Setattr((yyval.node),"name",(yyvsp[0].id));
	       }
#line 12209 "CParse/parser.c"
    break;

  case 615: /* kwargs: idstring COMMA kwargs  */
#line 7475 "./CParse/parser.y"
                                       {
                 (yyval.node) = NewHash();
                 Setattr((yyval.node),"name",(yyvsp[-2].id));
                 set_nextSibling((yyval.node),(yyvsp[0].node));
               }
#line 12219 "CParse/parser.c"
    break;

  case 616: /* kwargs: idstring EQUAL stringtype  */
#line 7480 "./CParse/parser.y"
                                            {
                 (yyval.node) = (yyvsp[0].node);
		 Setattr((yyval.node),"name",(yyvsp[-2].id));
               }
#line 12228 "CParse/parser.c"
    break;

  case 617: /* kwargs: idstring EQUAL stringtype COMMA kwargs  */
#line 7484 "./CParse/parser.y"
                                                        {
                 (yyval.node) = (yyvsp[-2].node);
		 Setattr((yyval.node),"name",(yyvsp[-4].id));
		 set_nextSibling((yyval.node),(yyvsp[0].node));
               }
#line 12238 "CParse/parser.c"
    break;

  case 619: /* stringnum: exprnum  */
#line 7492 "./CParse/parser.y"
                         {
                 (yyval.str) = Char((yyvsp[0].dtype).val);
               }
#line 12246 "CParse/parser.c"
    break;


#line 12250 "CParse/parser.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= END)
        {
          /* Return failure if at end of input.  */
          if (yychar == END)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 7497 "./CParse/parser.y"


SwigType *Swig_cparse_type(String *s) {
   String *ns;
   ns = NewString(s);
   Seek(ns,0,SEEK_SET);
   scanner_file(ns);
   top = 0;
   scanner_next_token(PARSETYPE);
   yyparse();
   /*   Printf(stdout,"typeparse: '%s' ---> '%s'\n", s, top); */
   return (SwigType *)top;
}


Parm *Swig_cparse_parm(String *s) {
   String *ns;
   ns = NewString(s);
   Seek(ns,0,SEEK_SET);
   scanner_file(ns);
   top = 0;
   scanner_next_token(PARSEPARM);
   yyparse();
   /*   Printf(stdout,"parmparse: '%s' ---> '%s'\n", s, top); */
   Delete(ns);
   return (Parm *)top;
}


ParmList *Swig_cparse_parms(String *s, Node *file_line_node) {
   String *ns;
   char *cs = Char(s);
   if (cs && cs[0] != '(') {
     ns = NewStringf("(%s)",s);
   } else {
     ns = NewString(s);
   }
   Setfile(ns, Getfile(file_line_node));
   Setline(ns, Getline(file_line_node));
   Seek(ns,0,SEEK_SET);
   scanner_file(ns);
   top = 0;
   scanner_next_token(PARSEPARMS);
   yyparse();
   /*   Printf(stdout,"parmsparse: '%s' ---> '%s'\n", s, top); */
   return (ParmList *)top;
}

